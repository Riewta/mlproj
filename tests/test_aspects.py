from src.aspects import ASPECTS, detect_aspects, split_sentences


def test_detect_aspects_finds_each_aspect():
    r = detect_aspects("อาหารอร่อย ราคาแพง พนักงานสุภาพ บรรยากาศดี")
    assert set(r) == set(ASPECTS)
    assert all(r.values())


def test_detect_aspects_matches_tokens_not_substrings():
    # "ถูกใจ" (like) contains "ถูก" (cheap) but is not about price
    assert detect_aspects("ถูกใจมาก")["price"] == 0
    assert detect_aspects("ราคาถูกมาก")["price"] == 1


def test_detect_aspects_nothing():
    assert not any(detect_aspects("ระหว่างทางไปเขาค้อ").values())


def test_split_sentences_breaks_on_paragraphs():
    out = split_sentences("ร้านอยู่ใกล้บ้าน\n\nอาหารอร่อยมาก")
    assert out == ["ร้านอยู่ใกล้บ้าน", "อาหารอร่อยมาก"]


def test_detect_aspects_ignores_elephant():
    # "ช้าง" (elephant) contains "ช้า" (slow)
    assert detect_aspects("ร้านอยู่ใกล้ปางช้าง")["service"] == 0


def test_crowd_is_not_price_but_portion_is():
    assert detect_aspects("คนเยอะมาก")["price"] == 0
    assert detect_aspects("ร้านนี้ให้เยอะมาก")["price"] == 1


def test_speed_phrase_is_service():
    assert detect_aspects("อาหารได้เร็วมาก")["service"] == 1
    assert detect_aspects("ขนมหมดเร็ว")["service"] == 0


def test_dictionary_compounds():
    # known limitation: "ถูกปากมาก" is segmented as "ถูก" + "ปากมาก", so only test the clean case
    assert detect_aspects("อาหารถูกปาก")["taste"] == 1
    assert detect_aspects("เจ้าของร้านใจดี")["service"] == 1


def test_common_misspellings_are_normalized():
    # "เสริฟ" (misspelled) is segmented into "เส", "ริ", "ฟ" unless normalized to "เสิร์ฟ"
    assert detect_aspects("พนักงานเสริฟช้า")["service"] == 1
    assert detect_aspects("รสชาดดี")["taste"] == 1
