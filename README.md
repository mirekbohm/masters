# Optimalizační metody portfolia
## Bc. Miroslav Böhm

## Popis
Podkladový kód pro diplomovou práci s názvem Optimalizační metody portfolia, kde dochází ke zkoumání vlivu korelační struktury portfolia na výsledky optimalizačních metod portfolia.

## Instalace prostředí a spuštění kódu
Tato práce byla zpracována pomocí Python manageru s názvem uv, který usnadňuje práci s virtuálním prostředím Pythonu. Pro potřeby spuštění kódu však stačí nainstalovat Python klasickou cestou ze stránek [Pythonu](https://www.python.org/downloads/).
### Instalace editoru, git a Python prostředí
1. Stáhněte editor Visual Studio Code z oficiálních stránek: [(https://code.visualstudio.com/download)](https://code.visualstudio.com/download)
2. Nainstalujte editor podle pokynů instalačního programu.
3. Nainstalujte Git, který si můžete stáhnout a nainstalovat z oficiálních stránek https://git-scm.com/downloads/win nebo následující příkazy stáhnou a nainstalují git automaticky:

```bash
# Na Windows - otevřete okno PowerShell a zkpoírujte následující příkaz.
winget install --id=Git.Git
```

```bash
# Na macOS nebo Linuxu proveďte následující příkaz v terminálu.
sudo apt install git-all
```

4. Pokud chcete nainstalovat uv:

```bash
# Na Windows - otevřete okno PowerShell a zkpoírujte následující příkaz.
winget install --id=astral-sh.uv
```

```bash
# Na macOS nebo Linuxu proveďte následující příkaz v terminálu.
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Spuštění kódu
1. V cílovém umístění (například na ploše) otevřete okno PowerShell a proveďte následující příkaz:

```bash
cd Desktop
git clone https://github.com/mirekbohm/masters.git
```

Tímto příkazem se stáhne zdrojový kód celého projektu do cílového umístění

2. Otevřete složku projektu (s názvem "masters") v editoru VS Code

```bash
code masters
```

3. Pokud už máte Python nainstalovaný klasickou cestou, tak stačí nainstalovat potřebné knihovny pro tento projekt:

```bash
cd masters
pip install -r requirements.txt
```
4. Pokud máte nainstalovaný uv, tak v lokaci projektu spusťte:

```bash
cd masters
uv sync
```

Tímto se připraví virtuální prostředí uv a kód je připraven ke spuštění v editoru. Veškerý kód je ve složce src.

