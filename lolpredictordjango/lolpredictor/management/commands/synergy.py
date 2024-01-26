import argparse

from django.core.management.base import BaseCommand

from synergy.cli import main


class Command(BaseCommand):
    help = "Run the synergy data and model pipeline (see `python -m synergy --help`)"

    def add_arguments(self, parser):
        parser.add_argument("pipeline_args", nargs=argparse.REMAINDER)

    def handle(self, *args, **options):
        raise SystemExit(main(options["pipeline_args"]))
