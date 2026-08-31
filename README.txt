MP3 → SRT
=========

1. Otevřete index.html v aktuálním Edge nebo Chrome, ideálně přes GitHub Pages / HTTPS.
2. Vyberte MP3.
3. Pro slovenštinu ponechte jazyk „Slovenština“ a model „Base“.
4. Výpočet ponechte na „Automaticky – GPU pokud jde“.
5. Klikněte na „Vytvořit titulky“.
6. Po dokončení stáhněte SRT nebo TXT.

Důležité:
- MP3 zůstává v počítači. Stahuje se pouze knihovna Transformers.js a model Whisper.
- První spuštění vyžaduje internet a stažení modelu; další spuštění používají cache prohlížeče.
- Automatický režim použije WebGPU, pokud je dostupné, jinak WASM/CPU.
- Když WebGPU selže během dlouhého přepisu, hotový text se nezahodí. Aplikace zopakuje jen aktuální úsek přes WASM a pokračuje dál.
- Model Base je doporučený kompromis rychlosti a kvality. Tiny je rychlejší, Small přesnější, ale výrazně náročnější.
- Whisper je citlivý na silnou kvantizaci encoderu. Aplikace proto používá přesnější encoder a zmenšuje hlavně decoder.
- Přepis probíhá po přibližně 30sekundových úsecích s malým překryvem a výsledek se průběžně zobrazuje.
