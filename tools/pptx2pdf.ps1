param([string]$in, [string]$out)
$p = New-Object -ComObject PowerPoint.Application
try { $d = $p.Presentations.Open($in, $true, $false, $false); $d.SaveAs($out, 32); $d.Close() } finally { $p.Quit() }
