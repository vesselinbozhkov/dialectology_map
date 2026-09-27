# Диалектна карта на България

Цифровизация на **Българския диалектен атлас. Обобщаващ том** (ИБЕ–БАН; т. I–III 2001, т. IV 2016)
като база данни с координати. Целта е интерактивна обобщена диалектна карта, а по-нататък
инструмент за разпознаване на диалект в жива реч спрямо книжовната норма.

## Съдържание

| Път | Какво е |
|---|---|
| `docs/research-data-sources.md` | Проучване на източниците, достъп до електронната библиотека на ИБЕ |
| `docs/atlas-overview.md` | Какво представляват атласите, как са кодирани картите, предложение за база и приоритизиране |
| `docs/extraction-vol4.md` | Процесът на автоматично извличане от т. IV и неговите ограничения |
| `catalog/maps.csv` | Каталог на всички 513 карти (номер, заглавие от OCR, печатна страница, страница в сканирането) |
| `output/bda_vol4.gpkg` | **База данни (GeoPackage, WGS84)** с ареалите на 145-те карти от т. IV |
| `scripts/` | Сваляне на страниците, PDF с текстов слой, каталог |
| `pipeline/` | Подравняване, георефериране, сегментация, износ към GeoJSON/уеб преглед |

## Как се пуска

```sh
pip install -r requirements.txt           # + apt: tesseract-ocr tesseract-ocr-bul
python scripts/fetch_pages.py             # страниците → data/raw/ (мрежа до ibl.bas.bg)
python scripts/build_pdfs.py              # PDF с OCR → data/pdf/
python scripts/build_catalog.py           # catalog/maps.csv
python pipeline/run_vol4.py               # т. IV → data/out/bda_vol4.gpkg
python pipeline/export_web.py             # GeoJSON + преглед → data/out/web/
```

Папката `data/` не се пази в git (сканове, OCR, междинни файлове). Данните са за лична употреба.
