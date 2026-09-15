import unittest
from recommender import RecommendationEngine, Resource, UserProfile


class RecommendationEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = RecommendationEngine([
            Resource("1", "Python ML", ("python", "machine learning"), "beginner", "course", "hands-on", 4.8),
            Resource("2", "SQL Basics", ("sql", "databases"), "beginner", "course", "hands-on", 4.9),
            Resource("3", "Advanced DL", ("deep learning", "python"), "advanced", "book", "theory", 4.7),
        ])

    def test_interest_match_ranks_first(self):
        profile = UserProfile(("machine learning",), "beginner", "course", "hands-on")
        result = self.engine.recommend(profile, top_k=1)
        self.assertEqual(result[0]["title"], "Python ML")

    def test_empty_profile_is_rejected(self):
        with self.assertRaises(ValueError):
            self.engine.recommend(UserProfile(tuple()), top_k=3)

    def test_top_k_is_respected(self):
        profile = UserProfile(("python",))
        self.assertEqual(len(self.engine.recommend(profile, top_k=2)), 2)


if __name__ == "__main__":
    unittest.main()
