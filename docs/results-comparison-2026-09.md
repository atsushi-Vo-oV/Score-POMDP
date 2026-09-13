# Comparison tables (generated 2026-09-14 JST)

Values: mean-chain evaluation return (10 episodes). `*` = still running (value at the latest evaluation). `fail` = terminated with a non-finite gradient. `--` = not run.

Reference values (scripted policies, Mountain Hike): stationary -360, observation-only -189, oracle (true state) -122.

### Mountain Hike, final evaluation (update 200)

| design | method | s10 | s11 | s12 | s13 | s14 | mean ± sd (complete seeds) |
|---|---|---|---|---|---|---|---|
| v1 (cold) | transformer | -443 | -445 | -182 | -447 | -263 | -356 ± 125 |
| v1 (cold) | alpha | -270 | -291 | -277 | -280 | -286 | -281 ± 8 |
| v1 (cold) | alpha-matched | -388* | -446* | -280* | -234* | -355* | -- |
| v1 (cold) | deepsets | -430 | -284 | -442 | -447 | -283 | -377 ± 86 |
| v2 (warm+anchor+aux) | transformer | -129 | -439 | -446 | -381 | -222 | -323 ± 141 |
| v2 (warm+anchor+aux) | alpha | -213 | -282 | -143 | -268 | -240 | -229 ± 55 |
| v2 (warm+anchor+aux) | alpha-matched | -261* | -443* | -271* | -238* | -254* | -- |
| v2 (warm+anchor+aux) | deepsets | -200 | -236 | -447 | -254 | -437 | -315 ± 118 |
| v3 (cold+aux) | transformer | -442 | -253 | -447 | -442 | -255 | -368 ± 104 |
| v3 (cold+aux) | alpha | -258 | -288 | -290 | -277 | -285 | -280 ± 13 |
| v3 (cold+aux) | alpha-matched | -444* | -444* | -429* | -444* | -350* | -- |
| v3 (cold+aux) | deepsets | -447 | -273 | -284 | -447 | -263 | -343 ± 95 |
| v4 (warm+anchor) | transformer | -277 | -285 | -276 | -279 | -227 | -269 ± 24 |
| v4 (warm+anchor) | alpha | -278 | -182 | -273 | -281 | -278 | -258 ± 43 |
| v4 (warm+anchor) | alpha-matched | -309* | -251* | -260* | -225* | -261* | -- |
| v4 (warm+anchor) | deepsets | -205 | -250 | -262 | -285 | -261 | -253 ± 30 |
| baseline | gru | -115 | -153 | -168 | -279 | -225 | -188 ± 65 |
| baseline | rnn | -128 | -162 | -132 | -202 | -277 | -180 ± 62 |
| baseline | pf | -165 | -162 | -129 | -146 | -126 | -146 ± 18 |
| baseline | pf-legacy | -447 | -201 | -279 | -- | -- | -309 ± 126 |

### Light-Dark 2D, final evaluation (update 200)

| design | method | s10 | s11 | s12 | s13 | s14 | mean ± sd (complete seeds) |
|---|---|---|---|---|---|---|---|
| v1 (cold) | transformer | -29 | -161 | -94 | -46 | -62 | -78 ± 52 |
| v1 (cold) | alpha | -158 | -179 | -378 | -91 | -125 | -186 ± 112 |
| v1 (cold) | alpha-matched | -70* | -216* | -113* | -1404* | -115* | -- |
| v1 (cold) | deepsets | -29 | -91 | -63 | -70 | -48 | -60 ± 23 |
| v2 (warm+anchor+aux) | transformer | -39 | -294 | -61 | -46 | -25 | -93 ± 113 |
| v2 (warm+anchor+aux) | alpha | -85 | -304 | -226 | -44 | -196 | -171 ± 106 |
| v2 (warm+anchor+aux) | alpha-matched | -888* | -464 | -115* | -189 | -489* | -326 ± 194 |
| v2 (warm+anchor+aux) | deepsets | -55 | -69 | -79 | -33 | -68 | -61 ± 18 |
| v3 (cold+aux) | transformer | -48 | -120 | -300 | -102 | -239 | -162 ± 104 |
| v3 (cold+aux) | alpha | -51 | -565 | -217 | -468 | -292 | -318 ± 204 |
| v3 (cold+aux) | alpha-matched | -160* | -201* | -1331* | -146* | -43* | -- |
| v3 (cold+aux) | deepsets | -23 | -209 | -96 | -371 | -140 | -168 ± 132 |
| v4 (warm+anchor) | transformer | -47 | -119 | -94 | -75 | -173 | -102 ± 48 |
| v4 (warm+anchor) | alpha | -106 | -437 | -168 | -101 | -373 | -237 ± 157 |
| v4 (warm+anchor) | alpha-matched | -296* | -460* | -297* | -459* | -444* | -- |
| v4 (warm+anchor) | deepsets | -63 | -125 | -246 | -43 | -141 | -124 ± 80 |
| baseline | gru | -29 | -104 | -70 | -69 | -70 | -68 ± 27 |
| baseline | rnn | -74 | -59 | -92 | -65 | -50 | -68 ± 16 |
| baseline | pf | fail | -53 | -72 | -55 | -57 | -59 ± 9 |

### Light-Dark 2D, mean of the last four evaluations (updates 125-200)

| design | method | s10 | s11 | s12 | s13 | s14 | mean ± sd (complete seeds) |
|---|---|---|---|---|---|---|---|
| v1 (cold) | transformer | -218 | -89 | -67 | -58 | -79 | -102 ± 66 |
| v1 (cold) | alpha | -266 | -166 | -201 | -322 | -207 | -232 ± 62 |
| v1 (cold) | alpha-matched | -184* | -521* | -366* | -853* | -436* | -- |
| v1 (cold) | deepsets | -228 | -57 | -116 | -72 | -45 | -104 ± 75 |
| v2 (warm+anchor+aux) | transformer | -110 | -181 | -260 | -126 | -134 | -162 ± 60 |
| v2 (warm+anchor+aux) | alpha | -152 | -305 | -150 | -230 | -563 | -280 ± 171 |
| v2 (warm+anchor+aux) | alpha-matched | -1647* | -867 | -310* | -735 | -554* | -801 ± 94 |
| v2 (warm+anchor+aux) | deepsets | -136 | -90 | -53 | -202 | -227 | -142 ± 73 |
| v3 (cold+aux) | transformer | -362 | -130 | -156 | -71 | -216 | -187 ± 111 |
| v3 (cold+aux) | alpha | -213 | -246 | -89 | -359 | -359 | -253 ± 113 |
| v3 (cold+aux) | alpha-matched | -361* | -394* | -1947* | -211* | -467* | -- |
| v3 (cold+aux) | deepsets | -199 | -176 | -86 | -419 | -220 | -220 ± 122 |
| v4 (warm+anchor) | transformer | -127 | -117 | -100 | -64 | -149 | -111 ± 32 |
| v4 (warm+anchor) | alpha | -272 | -429 | -142 | -365 | -290 | -300 ± 108 |
| v4 (warm+anchor) | alpha-matched | -331* | -271* | -539* | -1270* | -341* | -- |
| v4 (warm+anchor) | deepsets | -154 | -117 | -229 | -168 | -190 | -172 ± 42 |
| baseline | gru | -41 | -82 | -57 | -54 | -85 | -64 ± 19 |
| baseline | rnn | -78 | -55 | -70 | -47 | -57 | -62 ± 12 |
| baseline | pf | fail | -58 | -77 | -51 | -65 | -63 ± 11 |
