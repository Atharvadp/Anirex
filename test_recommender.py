from model.recommender import AnimeRecommender

print("Testing Anime Recommender...\n")

# Initialize recommender
rec = AnimeRecommender()

def _safe_get(res, path, default=None):
    cur = res
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

def run_case(case_name, *, genre, mood="", style="", reference_title=""):
    res = rec.get_recommendations(genre=genre, mood=mood, style=style, reference_title=reference_title)
    main_title = _safe_get(res, ["main", "title"])
    main_genre = _safe_get(res, ["main", "genre"])
    backups = res.get("backups") or []
    backup_titles = [b.get("title") for b in backups if isinstance(b, dict)]
    print(f"[{case_name}]")
    print(f"  inputs: genre={genre!r}, mood={mood!r}, style={style!r}, reference_title={reference_title!r}")
    print(f"  query_used: {res.get('query_used')!r}")
    print(f"  main: {main_title!r} | main_genre: {main_genre!r}")
    print(f"  backups: {backup_titles}")
    return res

def assert_true(label, condition):
    if condition:
        print(f"  PASS: {label}")
        return True
    print(f"  FAIL: {label}")
    return False

print("=== Smoke tests ===")
res1 = run_case("action-dark-modern-ref-aot", genre="Action", mood="Dark", style="Modern", reference_title="Attack on Titan")
assert_true("has main recommendation", bool(res1.get("main")))
print("")

res2 = run_case("slice-of-life-wholesome-modern", genre="Slice of Life", mood="Wholesome", style="Modern", reference_title="")
assert_true("has main recommendation", bool(res2.get("main")))
print("")

res3 = run_case("comedy-chill-classic-invalid-ref", genre="Comedy", mood="Chill", style="Classic", reference_title="NonExistentAnimeXYZ")
assert_true("returns at least 1 match", (res3.get("total_matches") or 0) >= 1)
print("")

print("=== Accuracy-ish sanity checks ===")
res4 = run_case("comedy-no-style", genre="Comedy", mood="", style="", reference_title="")
main_genre4 = _safe_get(res4, ["main", "genre"], "") or ""
assert_true("Comedy input yields a result tagged with Comedy",
            ("comedy" in str(main_genre4).lower()))
print("")

print("=== Same-series filtering checks ===")
res5 = run_case("gintama-comedy-ref", genre="Comedy", mood="", style="", reference_title="Gintama")
returned_titles5 = [t for t in [(_safe_get(res5, ["main", "title"]) or "")] + [b.get("title", "") for b in (res5.get("backups") or [])] if t]
assert_true("no returned title contains 'gintama'",
            all("gintama" not in str(t).lower() for t in returned_titles5))
print("")

print("=== Style influence checks (non-deterministic but should usually differ) ===")
res6a = run_case("action-dark-classic", genre="Action", mood="Dark", style="Classic", reference_title="")
res6b = run_case("action-dark-modern", genre="Action", mood="Dark", style="Modern", reference_title="")
titles6a = [(_safe_get(res6a, ["main", "title"]) or "")] + [b.get("title", "") for b in (res6a.get("backups") or [])]
titles6b = [(_safe_get(res6b, ["main", "title"]) or "")] + [b.get("title", "") for b in (res6b.get("backups") or [])]
assert_true("Classic vs Modern produce at least one different title",
            set(titles6a) != set(titles6b))
print("")

print("All tests executed.")
