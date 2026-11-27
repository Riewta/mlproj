# Aspect labeling guide

File: `data/labels/aspect_sentences.csv` (400 sentences from Wongnai reviews, made by `notebooks/05_aspects.ipynb`).

## What to fill

For each sentence, look at the four aspect columns. Write one of these in each column, or leave it **empty** if the sentence doesn't talk about that aspect:

| value | meaning | example |
|---|---|---|
| `pos` | praises this aspect | "น้ำซุปกลมกล่อม" → taste = pos |
| `neg` | criticizes this aspect | "รอนานเกือบชั่วโมง" → service = neg |
| `neu` | mentions it without praise or criticism | "ข้าวผัดจานละ 60 บาท" → price = neu |

A sentence can have **several aspects** ("อร่อยแต่แพง" → taste = pos, price = neg), or **none** (leave all four empty).
Use `note` for anything unclear, so the case can be discussed later. Don't skip a sentence.

Label **the sentence itself**. `context_before` / `context_after` are there only to help you understand it (for example, what "มัน" refers to). Don't label opinions that appear only in the context.

## The four aspects

| column | covers | does not cover |
|---|---|---|
| `taste` (อาหาร/รสชาติ) | taste, smell, texture, freshness, temperature of the food, how it looks, menu variety, drinks | the price of a dish |
| `price` (ราคา/ความคุ้มค่า) | price, value for money, portion size relative to price, promotions, service charge | whether the food is good |
| `service` (บริการ) | staff, owner, politeness, waiting time, speed, order mistakes, how complaints are handled | the room itself |
| `ambience` (บรรยากาศ/สถานที่) | decor, cleanliness, temperature/AC, noise, seating, view, toilets, photo spots | location, parking, how to get there |

## Edge cases (decided in advance)

- **Location, parking, directions, opening hours:** none of the four, so leave them empty. They are facts about the trip, not the restaurant experience.
- **Portion size:** "ให้เยอะ" / "ปริมาณน้อย" → `price` (value for money), not `taste`.
- **Waiting for food:** `service` = neg, even if the kitchen is the cause.
- **Cleanliness of the food** (hair in the dish): `taste` = neg. **Cleanliness of the place:** `ambience`.
- **Overall verdicts with no aspect:** "ประทับใจมาก", "จะมาอีก", "โดยรวมโอเค" → leave empty. They are sentiment, not an aspect.
- **Just naming a dish:** "สั่งข้าวมันไก่มา" → empty. **Naming a dish and judging it:** "ข้าวมันไก่อร่อย" → taste = pos.
- **Mild or hedged opinions:** "พอใช้ได้", "ธรรมดา", "ไม่ว้าว" → `neg` for that aspect. In Wongnai reviews this wording is how people complain politely (see notebook 02: these words push predictions toward negative).
- **Sarcasm:** label the meaning, not the words. "อร่อยมากจนต้องทิ้งครึ่งจาน" → taste = neg.
- **Comparisons:** "อร่อยกว่าสาขาอื่น" → taste = pos. "สู้ร้านเดิมไม่ได้" → neg.

## Tips

- Label the first 50, then reread this guide and fix anything you'd now label differently. Rules usually get clearer after the first batch.
- About 15 seconds per sentence: roughly 1.5–2 hours in total. Splitting it into several sessions helps you stay consistent.
- Save as **CSV UTF-8** (in Excel: "CSV UTF-8 (Comma delimited)"). Plain CSV in Excel can corrupt Thai text.

## Data license

The sentences come from [Wongnai reviews](https://huggingface.co/datasets/Wongnai/wongnai_reviews) (LGPL-3.0). This label file is redistributed under the same terms.
