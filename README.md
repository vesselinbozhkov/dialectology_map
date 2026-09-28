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
| `output/bda_vol1-3.gpkg` | Същото за 366-те карти от т. I–III (Ф 14 и Ф 15 липсват в онлайн сканирането) |
| `catalog/legends/` | Ръчно преписани заглавия и легенди с белег за книжовна норма (всичките 511 карти) |
| `scripts/` | Сваляне на страниците, PDF с текстов слой, каталог |
| `pipeline/` | Подравняване, георефериране, сегментация, износ към GeoJSON/уеб преглед |

## Как се пуска

```sh
pip install -r requirements.txt           # + apt: tesseract-ocr tesseract-ocr-bul
python scripts/fetch_pages.py             # страниците → data/raw/ (мрежа до ibl.bas.bg)
python scripts/build_pdfs.py              # PDF с OCR → data/pdf/
python scripts/build_catalog.py           # catalog/maps.csv
python pipeline/run_volume.py vol4        # т. IV → data/out/bda_vol4.gpkg
python pipeline/run_volume.py vol1-3      # т. I–III → data/out/bda_vol1-3.gpkg
python pipeline/verified.py vol4 && python pipeline/verified.py vol1-3  # ръчните легенди → колони title, text, grp, norm + таблица legend_extra
python pipeline/export_web.py vol4        # преглед → data/out/web/
python pipeline/export_web.py vol1-3      # преглед → data/out/web_vol1-3/
```

Папката `data/` не се пази в git (сканове, OCR, междинни файлове). Данните са за лична употреба.

## Онлайн прегледи

- Т. IV „Морфология“: https://claude.ai/artifact/S4tq5yuobruSjiumofKcTp
- Т. I–III „Фонетика, акцентология, лексика“: https://claude.ai/artifact/ETdMarcANr5YQSrRh1vZnY
