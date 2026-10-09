param([string]$in, [string]$out)
$w = New-Object -ComObject Word.Application
$w.Visible = $false
try { $d = $w.Documents.Open($in, $false, $true); $d.SaveAs2($out, 17); $d.Close($false) } finally { $w.Quit() }
