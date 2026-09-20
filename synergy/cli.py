import argparse
import asyncio
import json
import logging
import sys

from .config import get_settings
from .features.build import build_tables
from .ingest.seeder import crawl
from .ingest.store import Store
from .ingest.synthetic import generate
from .ml.score import SynergyService
from .ml.train import train


BLOCKS = ("movement", "embedding", "orphans", "reaction", "priority", "tendency", "habit", "hinge")


def _report(payload) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _build_blocks(settings, only: list[str] | None) -> dict:
    from .features.habit import build_habits
    from .features.hinge import build_hinge
    from .features.orphans import build_orphan_features
    from .features.priority import build_priority
    from .features.reaction import build_reaction
    from .features.tendency import build_tendencies
    from .ml.embedding import embedding_features
    from .ml.movement import movement_features

    makers = {
        "movement": lambda: movement_features(settings),
        "embedding": lambda: embedding_features(settings),
        "orphans": lambda: build_orphan_features(settings),
        "reaction": lambda: build_reaction(settings),
        "tendency": lambda: build_tendencies(settings),
        "habit": lambda: build_habits(settings),
        "hinge": lambda: build_hinge(settings),
        "priority": lambda: build_priority(settings),
    }
    out = {}
    for name in only or BLOCKS:
        out[name] = makers[name]()
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="synergy")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    crawl_cmd = sub.add_parser("crawl", help="seeded crawl of ranked matches from the Riot API")
    crawl_cmd.add_argument("--riot-id", default=None)
    crawl_cmd.add_argument("--max-matches", type=int, default=None)
    crawl_cmd.add_argument("--max-players", type=int, default=None)
    crawl_cmd.add_argument("--leaderboard", action="store_true", help="seed from the apex ladders")
    crawl_cmd.add_argument("--since-days", type=int, default=None)
    crawl_cmd.add_argument("--min-lp", type=int, default=None)
    crawl_cmd.add_argument("--max-tier", default=None)
    crawl_cmd.add_argument("--no-discover", action="store_true")
    crawl_cmd.add_argument("--matches-per-player", type=int, default=None)
    crawl_cmd.add_argument("--max-requests", type=int, default=None)

    breadth_cmd = sub.add_parser(
        "breadth", help="queue every Master and above player who already has depth in a position"
    )
    breadth_cmd.add_argument("--min-games", type=int, default=10, help="games in one position to qualify")

    synth_cmd = sub.add_parser("synthetic", help="generate a synthetic match corpus")
    synth_cmd.add_argument("--matches", type=int, default=400)
    synth_cmd.add_argument("--players", type=int, default=120)
    synth_cmd.add_argument("--seed", type=int, default=7)

    window_cmd = sub.add_parser("window", help="cut raw timelines down to the first 15 minutes")
    window_cmd.add_argument("--minutes", type=int, default=15)
    window_cmd.add_argument("--workers", type=int, default=None)

    features_cmd = sub.add_parser(
        "features", help="extract participation and pair tables from the 15 minute windows"
    )
    features_cmd.add_argument("--workers", type=int, default=None)

    train_cmd = sub.add_parser("train", help="train the pair synergy model")
    train_cmd.add_argument("--min-games", type=int, default=None)

    sub.add_parser("sequences", help="encode timelines into per player game tensors")

    deep_cmd = sub.add_parser("deep-train", help="train the timeline encoder on the GPU")
    deep_cmd.add_argument("--epochs", type=int, default=400)
    deep_cmd.add_argument("--dim", type=int, default=128)
    deep_cmd.add_argument("--embed-dim", type=int, default=64)
    deep_cmd.add_argument("--layers", type=int, default=3)
    deep_cmd.add_argument("--players-per-batch", type=int, default=64)
    deep_cmd.add_argument("--adversary-strength", type=float, default=1.0)
    deep_cmd.add_argument("--identity-conditioning", action="store_true")
    deep_cmd.add_argument("--device", default=None)


    sub.add_parser("db-load", help="load the raw archive on disk into postgres")
    reingest_cmd = sub.add_parser("reingest", help="rewrite frames and events from the raw timelines")
    reingest_cmd.add_argument("--missing", action="store_true", help="only matches with no frames")
    reingest_cmd.add_argument("--workers", type=int, default=None)

    stream_cmd = sub.add_parser("stream", help="write the per match event sequence")
    stream_cmd.add_argument("--limit", type=int, default=None)
    walk_cmd = sub.add_parser("walk-train", help="train the match walk on the event sequence")
    walk_cmd.add_argument("--epochs", type=int, default=10)
    walk_cmd.add_argument("--batch", type=int, default=128)
    blocks_cmd = sub.add_parser("blocks", help="build the derived feature tables the model loads")
    blocks_cmd.add_argument(
        "--only", nargs="*", choices=list(BLOCKS), default=None, help="build a subset"
    )
    variance_cmd = sub.add_parser(
        "variance", help="fit the pair identity variance component on advantage at 15"
    )
    variance_cmd.add_argument("--min-games", type=int, default=5)
    interact_cmd = sub.add_parser(
        "interaction", help="does a pair term over in match behaviour beat its null"
    )
    interact_cmd.add_argument("--rank", type=int, default=8)
    interact_cmd.add_argument("--steps", type=int, default=8000)
    interact_cmd.add_argument("--nulls", type=int, default=40)
    interact_cmd.add_argument(
        "--min-steps", type=int, default=0, help="run every rung at least this many steps"
    )
    interact_cmd.add_argument("--source", choices=("style", "walk"), default="style")
    interact_cmd.add_argument(
        "--matchup", action="store_true", help="add a rung for the term across the net"
    )
    interact_cmd.add_argument(
        "--scores", action="store_true", help="write per player styles and every corpus pair score"
    )
    interact_cmd.add_argument(
        "--skip-fit", action="store_true", help="reuse the saved matrix, only write the scores"
    )
    mirrored_cmd = sub.add_parser(
        "mirrored", help="fit the cells and their interaction against each pair's own gold at 15"
    )
    mirrored_cmd.add_argument("--components", type=int, default=24)
    mirrored_cmd.add_argument(
        "--positions-only", action="store_true", help="only refit the per position seat weights"
    )
    mirrored_cmd.add_argument(
        "--target",
        choices=("gold", "events"),
        default="gold",
        help="gold at 15, or the gold swung by events both players were present at",
    )
    pairnet_cmd = sub.add_parser(
        "pairnet", help="fit a network on the four seats of each pair against the pair's own gold at 15"
    )
    pairnet_cmd.add_argument("--epochs", type=int, default=12)
    pairnet_cmd.add_argument("--seeds", type=int, default=5, help="repeat the network and its null this many times")
    sub.add_parser("validate", help="check the corpus for integrity problems")
    sub.add_parser("profiles", help="rebuild the player profile table from the corpus")
    sub.add_parser("status", help="show corpus and model status")

    pair_cmd = sub.add_parser("pair", help="score one pairing, each player in the position they will play")
    pair_cmd.add_argument("left")
    pair_cmd.add_argument("right")
    pair_cmd.add_argument("--left-position", default=None, help="top, jungle, mid, bot or support")
    pair_cmd.add_argument("--right-position", default=None, help="top, jungle, mid, bot or support")

    lineup_cmd = sub.add_parser("lineup", help="score a full five, one player per position")
    for position in ("top", "jungle", "mid", "bot", "support"):
        lineup_cmd.add_argument(f"--{position}", required=True)

    partners_cmd = sub.add_parser("partners", help="best and worst partners for a player")
    partners_cmd.add_argument("player")
    partners_cmd.add_argument("--limit", type=int, default=10)

    ranks_cmd = sub.add_parser("ranks", help="look up rank for players that have none")
    ranks_cmd.add_argument("--min-games", type=int, default=1)
    ranks_cmd.add_argument("--max-requests", type=int, default=None)

    priors_cmd = sub.add_parser("priors", help="conditional base rates for a behaviour given the matchup")
    priors_cmd.add_argument("--json", action="store_true")
    priors_cmd.add_argument("--propensities", action="store_true")
    priors_cmd.add_argument("--players", default=None)
    priors_cmd.add_argument("--limit", type=int, default=10)

    all_cmd = sub.add_parser("all", help="synthetic corpus, features and training in one go")
    all_cmd.add_argument("--matches", type=int, default=400)
    all_cmd.add_argument("--players", type=int, default=120)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    settings = get_settings()
    settings.ensure_dirs()

    if args.command == "crawl":
        if args.max_matches:
            settings.max_matches = args.max_matches
        if args.max_players:
            settings.max_players = args.max_players
        if args.since_days is not None:
            settings.crawl_since_days = args.since_days
        if args.min_lp is not None:
            settings.apex_min_league_points = args.min_lp
        if args.max_tier:
            settings.max_tier = args.max_tier
        if args.no_discover:
            settings.crawl_discover = False
        if args.matches_per_player is not None:
            settings.matches_per_player = args.matches_per_player
        if args.max_requests:
            settings.max_requests = args.max_requests
        if not settings.riot_api_key:
            print("RIOT_API_KEY is not set", file=sys.stderr)
            return 2
        _report(asyncio.run(crawl(args.riot_id, settings, leaderboard=args.leaderboard)).as_dict())
        return 0

    if args.command == "breadth":
        from .ingest.breadth import enqueue

        _report(enqueue(settings, min_games=args.min_games))
        return 0

    if args.command == "synthetic":
        _report(generate(matches=args.matches, players=args.players, seed=args.seed, settings=settings))
        return 0

    if args.command == "window":
        from .ingest.window import build_windows

        _report(build_windows(settings, minutes=args.minutes, workers=args.workers))
        return 0

    if args.command == "features":
        tables = build_tables(settings, workers=args.workers)
        _report({
            key: value if isinstance(value, int) else len(value) for key, value in tables.items()
        })
        return 0

    if args.command == "reingest":
        from .db.reingest import reingest

        _report(reingest(settings, workers=args.workers, missing=args.missing))
        return 0

    if args.command == "ranks":
        from .ingest.seeder import refresh_ranks

        if args.max_requests:
            settings.max_requests = args.max_requests
        if not settings.riot_api_key:
            print("RIOT_API_KEY is not set", file=sys.stderr)
            return 2
        _report(asyncio.run(refresh_ranks(settings, min_games=args.min_games)))
        return 0

    if args.command == "priors":
        from .features.priors import propensity_priors, push_priors, render, top_players

        if args.propensities:
            _report(propensity_priors(settings))
            return 0
        if args.players:
            print(top_players(args.players, limit=args.limit, settings=settings).to_string())
            return 0
        payload = push_priors(settings)
        _report(payload) if args.json else print(render(payload))
        return 0

    if args.command == "train":
        _report(train(settings, min_games=args.min_games))
        return 0

    if args.command == "sequences":
        from .deep.sequences import build_sequences

        _report(build_sequences(settings))
        return 0

    if args.command == "deep-train":
        from .deep.train import train_encoder

        _report(
            train_encoder(
                settings,
                epochs=args.epochs,
                dim=args.dim,
                embed_dim=args.embed_dim,
                layers=args.layers,
                players_per_batch=args.players_per_batch,
                adversary_strength=args.adversary_strength,
                identity_conditioning=args.identity_conditioning,
                device=args.device,
            )
        )
        return 0

    if args.command == "db-load":
        from .db.load import load_from_archive

        _report(load_from_archive(settings))
        return 0

    if args.command == "stream":
        from .deep.stream import build_stream

        _report(build_stream(settings, limit=args.limit))
        return 0
    if args.command == "walk-train":
        from .deep.walk_train import train_walk

        report = train_walk(settings, epochs=args.epochs, batch=args.batch)
        _report(report["best"])
        return 0
    if args.command == "blocks":
        _report(_build_blocks(settings, args.only))
        return 0
    if args.command == "variance":
        from .features.build import load_tables
        from .ml.dataset import attach_dyads

        from .ml.gold import fit_gold

        tables = load_tables(settings, names=("pairs", "dyads"))
        pairs = attach_dyads(tables["pairs"], tables.get("dyads"))
        _report(fit_gold(pairs, settings, min_games=args.min_games))
        return 0
    if args.command == "pairnet":
        from .ml.pairnet import fit_pairnet

        _report(fit_pairnet(settings, epochs=args.epochs, seeds=args.seeds))
        return 0
    if args.command == "mirrored":
        from .ml.mirrored import fit_mirrored, fit_positions

        if args.target == "gold":
            _report(fit_positions(settings))
        if not args.positions_only:
            _report(fit_mirrored(settings, components=args.components, target=args.target))
        return 0
    if args.command == "interaction":
        from .ml.interaction import fit_interaction, write_scores

        if not args.skip_fit:
            _report(
                fit_interaction(
                    settings,
                    rank=args.rank,
                    steps=args.steps,
                    nulls=args.nulls,
                    source=args.source,
                    min_steps=args.min_steps,
                    matchup=args.matchup,
                )
            )
        if args.scores or args.skip_fit:
            _report(write_scores(settings, source=args.source))
        return 0
    if args.command == "validate":
        from .db.validate import validate

        report = validate(settings)
        _report(report)
        return 0 if not any(report.values()) else 1
    if args.command == "profiles":
        from .db.profiles import build_profiles as build_player_profiles

        _report(build_player_profiles(settings))
        return 0
    if args.command == "status":
        with Store(settings) as store:
            corpus = store.counts()
            corpus["frontier"] = store.frontier_counts()
            corpus["unnamed_frequent_players"] = len(store.unnamed_players(settings.min_profile_games))
        _report(
            {
                "corpus": corpus,
                "crawl_budget": settings.max_requests,
                "service": SynergyService(settings).load().status(),
            }
        )
        return 0

    if args.command == "pair":
        service = SynergyService(settings).load()
        _report(service.pair_score(args.left, args.right, args.left_position, args.right_position))
        return 0

    if args.command == "lineup":
        service = SynergyService(settings).load()
        _report(service.lineup({position: getattr(args, position) for position in ("top", "jungle", "mid", "bot", "support")}))
        return 0

    if args.command == "partners":
        service = SynergyService(settings).load()
        _report(service.best_partners(args.player, limit=args.limit))
        return 0

    if args.command == "all":
        from .ingest.window import build_windows

        generate(matches=args.matches, players=args.players, settings=settings)
        build_windows(settings)
        build_tables(settings)
        _report(train(settings))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
