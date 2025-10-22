# Установка **uv** под Windows

## Используйте встроенный установщик

Запустите PowerShell и в командной строке наберите:

```
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## После установки перезапустите терминал или обновите переменные среды

```
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
```

# Запуск приложения

В каталоге с приложением выполните в командной строке

```
uv run main.py
```