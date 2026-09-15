from recommender import UserProfile, default_engine


def choose(prompt: str, options: list[str], optional: bool = True) -> str | None:
    print("\n" + prompt)
    for i, option in enumerate(options, 1):
        print(f"  {i}. {option}")
    if optional:
        print("  0. No preference")

    while True:
        value = input("Choose: ").strip()
        if optional and value == "0":
            return None
        if value.isdigit() and 1 <= int(value) <= len(options):
            return options[int(value) - 1]
        print("Invalid choice. Try again.")


def main() -> None:
    print("=" * 62)
    print("AI LEARNING RESOURCE RECOMMENDER")
    print("DecodeLabs - Artificial Intelligence Project 3")
    print("=" * 62)

    interests_raw = input(
        "\nEnter your interests separated by commas\n"
        "(examples: python, machine learning, recommendation systems): "
    )
    interests = tuple(x.strip() for x in interests_raw.split(",") if x.strip())

    difficulty = choose("Preferred difficulty:", ["beginner", "intermediate", "advanced"])
    format_ = choose("Preferred format:", ["course", "tutorial", "project", "article", "book"])
    style = choose("Preferred learning style:", ["hands-on", "balanced", "theory", "visual"])

    profile = UserProfile(
        interests=interests,
        difficulty=difficulty,
        format=format_,
        style=style,
    )

    engine = default_engine()
    try:
        results = engine.recommend(profile, top_k=5)
    except ValueError as exc:
        print(f"\nError: {exc}")
        return

    print("\n" + "=" * 62)
    print("TOP RECOMMENDATIONS")
    print("=" * 62)

    for rank, item in enumerate(results, 1):
        print(f"\n{rank}. {item['title']}  |  Match: {item['match_percent']}%")
        print(f"   Difficulty: {item['difficulty']} | Format: {item['format']} | Style: {item['style']}")
        print(f"   Topics: {', '.join(item['topics'])}")
        print(f"   Why: {'; '.join(item['reasons'])}")


if __name__ == "__main__":
    main()
