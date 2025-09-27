## Option 5 – Household Matching (Plain‑English Guide)

This guide explains, in simple terms, how the Option 5 matching works and what results mean. It is written for business users, analysts, and non‑technical stakeholders.

### What Option 5 is for
Option 5 answers a practical question: “For each household in Table 1, how much of that same household can we find in Table 2?”

It measures coverage from the Table 1 point of view. If a large portion of a Table 1 household’s members are also found in Table 2, that is a strong signal it’s the same household.

### The fields we need
For each person in both tables, we need:
- Household ID
  - Table 1 (source) uses a household ID called uuid (a 36‑character identifier).
  - Table 2 (comparison) uses its own household ID (internally the record id is used to identify the household group in Option 5).
- Person’s first name and last name
- Person’s birthdate

Notes:
- Middle name is optional and is not used by Option 5 matching.
- Birthdate is required for a match. If the birthdate differs or is missing, that person will not match under Option 5.

### How Option 5 decides a person‑to‑person match
Option 5 looks for two conditions:
1) Birthdate must be exactly the same in both tables.
2) Names must be similar (first and last name). The system allows minor spelling differences or variations, but they must be close enough to count.

If both are true, we treat those two person records as a match.

### How Option 5 turns person matches into household matches
After finding matching people, Option 5 summarizes at the household level:
- Table 1 households are grouped by their uuid.
- Table 2 households are grouped by their own household identifier (internally it is the id field).
- For each person in a Table 1 household, the system assigns them to the single “best” matching household in Table 2 (if there’s a tie between multiple Table 2 households, that person is not counted to avoid double‑counting).
- For each pair (Table 1 household, Table 2 household), we count how many unique people from the Table 1 household matched.

### The household match percentage (what 100% means)
Option 5 calculates:
- Household match percentage = (Number of matched people from the Table 1 household) divided by (Total people in that Table 1 household), expressed as a percentage.
- Only results strictly greater than 50% are kept. Exactly 50% is not included.

Important: The denominator is always the size of the household in Table 1. This means the percentage answers the question: “How much of the Table 1 household did we find in Table 2?” It does not penalize extra, unmatched people that exist only in Table 2.

### Why a 1‑person household can show 100%
If a Table 1 household has only 1 member and that person is found in Table 2, then the calculation is 1 ÷ 1 = 100%. This is expected and correct because the metric is measuring coverage of Table 1, not a two‑way overlap.

### Practical examples
- Example A: T1 household has 1 member
  - 1 of 1 member matches in T2 → 100% → included (because > 50%).
- Example B: T1 household has 2 members
  - 1 of 2 members matches → 50% → not included (the rule is strictly greater than 50%).
  - 2 of 2 members match → 100% → included.
- Example C: T1 household has 3 members
  - 1 of 3 members matches → 33% → not included.
  - 2 of 3 members match → 67% → included.
- Example D: T1 household has 5 members
  - 3 of 5 members match → 60% → included.

### What Option 5 does not do
- It does not average sizes across tables. The denominator is always the Table 1 household size.
- It does not reduce a percentage because Table 2 has extra, unmatched people.
- It does not use middle names in the similarity check.
- It does not match when birthdates differ.

### Why we designed it this way
- Clear business signal: We want to know whether “most” of a Table 1 household is present in Table 2. Using Table 1’s size as the denominator directly answers that.
- Data quality resilience: Requiring the same birthdate avoids accidental matches on common names. Allowing “similar” names captures small spelling differences and other small variations.
- Conservative counting: Assigning each Table 1 person to only one best matching Table 2 household prevents inflating counts by double‑counting the same person across multiple households.
- Focus on strong matches: Keeping only results above 50% highlights likely true household matches and reduces noise.

### Common questions
- Q: Why do some household results show 100% even when Table 2 has more people?
  - A: Because 100% means “we found all of the Table 1 household in Table 2.” Extra people that only appear in Table 2 do not reduce this percentage.

- Q: Can we include exact 50% matches?
  - A: The current rule is strictly greater than 50%. If you need to include exactly 50%, this can be changed, but it would broaden the result set.

- Q: Can we also penalize extra people in Table 2?
  - A: That would require a different measure (for example, using both Table 1 and Table 2 sizes). Option 5 is intentionally one‑sided (Table 1 coverage). If you need a symmetric measure, we can propose alternatives.

### Data quality tips for reliable results
- Use full household IDs for Table 1 (complete 36‑character uuid). If each person has a unique household ID, you will naturally see many 100% results because each “household” has only one member.
- Make sure birthdates are correct. Option 5 will not match people if their birthdates do not exactly match.
- Expect a mix of results. Households with more members should produce varied percentages (for example, 60%, 67%, 75%, 100%), depending on how many members match.

### Quick recap
- Option 5 tells you how much of a Table 1 household appears in Table 2.
- A person matches only if birthdate is identical and names are similar.
- Results are kept only when more than 50% of the Table 1 household is found.
- The percentage is from the Table 1 perspective, so extra members in Table 2 don’t lower it.

