from contextlib import contextmanager

import psycopg

from ..config import Settings

CORPUS_LOCK = 72_100_427
PREFIX = "synergy "


class CorpusBusy(RuntimeError):
    pass


def _holder(connection) -> str:
    row = connection.execute(
        "SELECT a.application_name FROM pg_locks l JOIN pg_stat_activity a ON a.pid = l.pid"
        " WHERE l.locktype = 'advisory' AND l.classid = 0 AND l.objid = %s AND l.granted LIMIT 1",
        (CORPUS_LOCK,),
    ).fetchone()
    return row[0].removeprefix(PREFIX) if row and row[0] else "another job"


@contextmanager
def corpus_lock(settings: Settings, holder: str):
    connection = psycopg.connect(settings.database_url, autocommit=True, application_name=f"{PREFIX}{holder}")
    try:
        if not connection.execute("SELECT pg_try_advisory_lock(%s)", (CORPUS_LOCK,)).fetchone()[0]:
            raise CorpusBusy(f"The corpus is busy with {_holder(connection)}. Try again when it finishes.")
        yield
    finally:
        connection.close()
