## What’s improved in SRS-II Name Matching Application

We’ve made name matching more reliable, especially for names with accents and characters from different languages.

What changed in simple terms:
- Prevented rare crashes when a name contained unusual characters (for example, symbols or non‑Latin letters). The app now handles these safely.
- Improved handling of accented letters (like José, Müller). These are recognized more accurately during matching.
- No changes to your workflow — everything works the same, just more robust.

Why this matters:
- Fewer failures during large matching runs — better stability and fewer interruptions.
- Better match quality for international data — reduces missed matches caused by accents or special characters.
- Safer processing — the system guards against edge cases so matching completes as expected.

Notes:
- This change focuses on improving reliability for phonetic matching in English‑like names, while keeping general fuzzy matching for all languages.
- There are no changes to exports or file formats.



## New: Match Across Two Databases

You can now compare and match names between two different MySQL databases or servers in one run.

What this means for you:
- Connect to two sources (e.g., Production and Archive) and find matches directly across them.
- Large lists still run efficiently — the app streams data and recovers from interruptions.
- No changes to how you export results — CSV and Excel outputs continue to work.

How to use (simple):
- Keep using the app the same as before for one database.
- To enable two-database matching, set these environment variables for the second database:
  - DB2_HOST, DB2_PORT, DB2_USER, DB2_PASS, DB2_DATABASE
- If these are set, the app automatically connects to both and matches across them.

Why this matters:
- Saves time by removing manual steps to move data between servers.
- Reduces errors: you see cross-database matches in one place.
- Scales for big jobs with the same stability improvements.
