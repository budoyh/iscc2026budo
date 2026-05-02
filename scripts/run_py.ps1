param(
  [Parameter(Mandatory=$true, ValueFromRemainingArguments=$true)]
  [string[]]$Args
)

$root = Split-Path -Parent $PSScriptRoot
$python = "C:\Users\Asus\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$deps = Join-Path $root ".deps\py312"

$env:PYTHONPATH = $deps
& $python @Args
