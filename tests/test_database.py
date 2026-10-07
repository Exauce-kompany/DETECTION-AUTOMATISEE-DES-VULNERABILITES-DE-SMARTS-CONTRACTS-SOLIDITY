"""History schema compatibility, rollback and connection lifetime."""

import importlib
import sqlite3
import unittest
from unittest.mock import patch

from webapp.persistence import database


class DatabaseTests(unittest.TestCase):
    def test_importing_application_does_not_create_database(self):
        with patch.object(database, "init_database") as initialize:
            import webapp.app

            importlib.reload(webapp.app)
        initialize.assert_not_called()

    def test_failed_transaction_rolls_back_and_closes_connection(self):
        connection = sqlite3.connect(":memory:")
        with patch.object(database, "get_connection", return_value=connection):
            with self.assertRaisesRegex(ValueError, "rollback"):
                with database.transaction() as active:
                    active.execute("CREATE TABLE example (value INTEGER)")
                    active.execute("INSERT INTO example VALUES (1)")
                    raise ValueError("rollback")
        with self.assertRaises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")

    def test_history_crud_and_statistics(self):
        # Keep a named in-memory database alive across per-operation connections.
        uri = "file:history-test?mode=memory&cache=shared"
        keeper = sqlite3.connect(uri, uri=True)
        self.addCleanup(keeper.close)

        def connect():
            connection = sqlite3.connect(uri, uri=True)
            connection.row_factory = sqlite3.Row
            return connection

        with patch.object(database, "get_connection", side_effect=connect):
            database.init_database()
            self.assertEqual(database.get_analysis_statistics()["total_analyses"], 0)
            identifier = database.save_analysis(
                "c.sol",
                13,
                1,
                "signal",
                90,
                90,
                10,
                90,
                "critique",
                3,
                3,
                0,
                0,
                False,
                "contract C {}",
            )
            self.assertEqual(database.get_analysis_by_id(identifier)["code"], "contract C {}")
            self.assertEqual(len(database.get_all_analyses()), 1)
            self.assertEqual(
                database.get_analysis_statistics(),
                {
                    "total_analyses": 1,
                    "vulnerable": 1,
                    "non_vulnerable": 0,
                    "average_confidence": 90,
                },
            )
            self.assertTrue(database.delete_analysis(identifier))
            self.assertFalse(database.delete_analysis(identifier))
            self.assertIsNone(database.get_analysis_by_id(identifier))
            database.clear_history()
