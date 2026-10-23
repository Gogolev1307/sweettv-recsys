# Recommendation examples

All models use the selected configurations from validation_methods.csv. Seed: 42; watch threshold: 300 seconds.
The heaviest viewer maximises the number of watched A airings among all 1,260 test users. The lightest viewer minimises that count among test users with at least one A watched airing and nonempty B truth. User-id ties are broken in ascending order. The top 10 history programmes are ordered by A seconds, then programme id.
Titles/categories use the latest A metadata, with earliest B metadata as a fallback. A-history flags and distinct-viewer popularity use A only; relevance and AP use B only.

## Heaviest viewer: `13571199046623737884`

History: **2,022 watched A airings**, **649 distinct A programmes**, **9,388,931 watched A seconds**. B truth: **259 distinct programmes**.

### Top programmes watched in A

| Programme id | Title | Category | A seconds |
|---:|---|---|---:|
| 2400480 | ТСН. Неделя. | Инфо | 892,255 |
| 20088 | 'Завтрак с 1+1'. Информационно-развлекательная программа. | Развлечения | 350,139 |
| 240081 | Факты. День. | Инфо | 252,975 |
| 2400508 | Сегодня. Итоги с Олегом Панютой. | Инфо | 245,517 |
| 6500479 | 'Мир наизнанку'. | Познавательное | 244,966 |
| 10002451 | Т/с 'Великолепный век', 81 и 82 с. | Сериалы | 232,173 |
| 10003159 | Т/с 'Ничто не случается дважды', 8 с. | Сериалы | 218,222 |
| 2400467 | 'Викна-Новости'. | Инфо | 196,684 |
| 500315 | Ток-шоу 'Касается каждого'. | Развлечения | 153,138 |
| 400357 | 'Холостяк', 10 сезон, 11 эп. | Развлечения | 141,714 |

### popularity: AP@5 = 1.000000000

Configuration: `{}`

| Rank | Programme id | Title | Category | Distinct A viewers | Watched in A? | Relevant in B? |
|---:|---:|---|---|---:|:---:|:---:|
| 1 | 2400480 | ТСН. Неделя. | Инфо | 4,362 | yes | yes |
| 2 | 240081 | Факты. День. | Инфо | 4,062 | yes | yes |
| 3 | 6500479 | 'Мир наизнанку'. | Познавательное | 3,610 | yes | yes |
| 4 | 2400508 | Сегодня. Итоги с Олегом Панютой. | Инфо | 3,608 | yes | yes |
| 5 | 2400467 | 'Викна-Новости'. | Инфо | 3,538 | yes | yes |

### rewatch: AP@5 = 0.643333333

Configuration: `{}`

| Rank | Programme id | Title | Category | Distinct A viewers | Watched in A? | Relevant in B? |
|---:|---:|---|---|---:|:---:|:---:|
| 1 | 2400480 | ТСН. Неделя. | Инфо | 4,362 | yes | yes |
| 2 | 20088 | 'Завтрак с 1+1'. Информационно-развлекательная программа. | Развлечения | 2,696 | yes | no |
| 3 | 240081 | Факты. День. | Инфо | 4,062 | yes | yes |
| 4 | 2400508 | Сегодня. Итоги с Олегом Панютой. | Инфо | 3,608 | yes | yes |
| 5 | 6500479 | 'Мир наизнанку'. | Познавательное | 3,610 | yes | yes |

### MF: AP@5 = 0.353333333

Configuration: `{"alpha":1,"factors":4,"iterations":10,"reg":10,"seed":42}`

| Rank | Programme id | Title | Category | Distinct A viewers | Watched in A? | Relevant in B? |
|---:|---:|---|---|---:|:---:|:---:|
| 1 | 90077687 | Х/ф 'Великий уравнитель'. | Фильмы | 1,171 | yes | no |
| 2 | 2500430 | 'Затерянный мир'. | Познавательное | 2,506 | yes | yes |
| 3 | 90076295 | Х/ф 'Призрачный патруль'. | Фильмы | 1,016 | yes | yes |
| 4 | 90081867 | Х/ф 'Миссия невыполнима: Последствия'. | Фильмы | 892 | yes | no |
| 5 | 2500413 | Гражданская оборона. | Инфо | 2,348 | yes | yes |

### PPR: AP@5 = 1.000000000

Configuration: `{"alpha":0.8,"batch_size":128,"max_iterations":500,"tolerance":1e-06}`

| Rank | Programme id | Title | Category | Distinct A viewers | Watched in A? | Relevant in B? |
|---:|---:|---|---|---:|:---:|:---:|
| 1 | 2400480 | ТСН. Неделя. | Инфо | 4,362 | yes | yes |
| 2 | 240081 | Факты. День. | Инфо | 4,062 | yes | yes |
| 3 | 6500479 | 'Мир наизнанку'. | Познавательное | 3,610 | yes | yes |
| 4 | 2400508 | Сегодня. Итоги с Олегом Панютой. | Инфо | 3,608 | yes | yes |
| 5 | 2400467 | 'Викна-Новости'. | Инфо | 3,538 | yes | yes |

### CBF: AP@5 = 0.000000000

Configuration: `{"idf":"smooth_A_only","profile_weight":"log1p(seconds/60)","tf":"binary","year":"first_year_decade"}`

| Rank | Programme id | Title | Category | Distinct A viewers | Watched in A? | Relevant in B? |
|---:|---:|---|---|---:|:---:|:---:|
| 1 | 90077525 | Х/ф 'Пластик'. | Фильмы | 0 | no | no |
| 2 | 90080935 | Х/ф 'Уличный боец'. | Фильмы | 0 | no | no |
| 3 | 90073483 | Х/ф 'Легенда танцующего ниндзя'. | Фильмы | 0 | no | no |
| 4 | 90082800 | Х/ф 'Волшебная малина' | Фильмы | 0 | no | no |
| 5 | 90076148 | Х/ф 'Войны орков'. | Фильмы | 0 | no | no |

### association: AP@5 = 1.000000000

Configuration: `{"lift_strictly_greater_than":1,"minimum_support":20,"score":"max_confidence"}`

| Rank | Programme id | Title | Category | Distinct A viewers | Watched in A? | Relevant in B? |
|---:|---:|---|---|---:|:---:|:---:|
| 1 | 240081 | Факты. День. | Инфо | 4,062 | yes | yes |
| 2 | 2400480 | ТСН. Неделя. | Инфо | 4,362 | yes | yes |
| 3 | 2400508 | Сегодня. Итоги с Олегом Панютой. | Инфо | 3,608 | yes | yes |
| 4 | 2400467 | 'Викна-Новости'. | Инфо | 3,538 | yes | yes |
| 5 | 6500479 | 'Мир наизнанку'. | Познавательное | 3,610 | yes | yes |

### hybrid: AP@5 = 1.000000000

Configuration: `{"MF":2.0,"PPR":2.0,"popularity":0.0,"rewatch":0.5}`

| Rank | Programme id | Title | Category | Distinct A viewers | Watched in A? | Relevant in B? |
|---:|---:|---|---|---:|:---:|:---:|
| 1 | 2400480 | ТСН. Неделя. | Инфо | 4,362 | yes | yes |
| 2 | 240081 | Факты. День. | Инфо | 4,062 | yes | yes |
| 3 | 2400508 | Сегодня. Итоги с Олегом Панютой. | Инфо | 3,608 | yes | yes |
| 4 | 6500479 | 'Мир наизнанку'. | Познавательное | 3,610 | yes | yes |
| 5 | 2400467 | 'Викна-Новости'. | Инфо | 3,538 | yes | yes |

## Lightest eligible viewer: `8150816946177670084`

History: **1 watched A airings**, **1 distinct A programmes**, **2,753 watched A seconds**. B truth: **12 distinct programmes**.

### Top programmes watched in A

| Programme id | Title | Category | A seconds |
|---:|---|---|---:|
| 90053098 | Х/ф 'Мужество в бою'. | Фильмы | 2,753 |

### popularity: AP@5 = 0.100000000

Configuration: `{}`

| Rank | Programme id | Title | Category | Distinct A viewers | Watched in A? | Relevant in B? |
|---:|---:|---|---|---:|:---:|:---:|
| 1 | 2400480 | ТСН. Неделя. | Инфо | 4,362 | no | no |
| 2 | 240081 | Факты. День. | Инфо | 4,062 | no | yes |
| 3 | 6500479 | 'Мир наизнанку'. | Познавательное | 3,610 | no | no |
| 4 | 2400508 | Сегодня. Итоги с Олегом Панютой. | Инфо | 3,608 | no | no |
| 5 | 2400467 | 'Викна-Новости'. | Инфо | 3,538 | no | no |

### rewatch: AP@5 = 0.066666667

Configuration: `{}`

| Rank | Programme id | Title | Category | Distinct A viewers | Watched in A? | Relevant in B? |
|---:|---:|---|---|---:|:---:|:---:|
| 1 | 90053098 | Х/ф 'Мужество в бою'. | Фильмы | 221 | yes | no |
| 2 | 2400480 | ТСН. Неделя. | Инфо | 4,362 | no | no |
| 3 | 240081 | Факты. День. | Инфо | 4,062 | no | yes |
| 4 | 6500479 | 'Мир наизнанку'. | Познавательное | 3,610 | no | no |
| 5 | 2400508 | Сегодня. Итоги с Олегом Панютой. | Инфо | 3,608 | no | no |

### MF: AP@5 = 0.000000000

Configuration: `{"alpha":1,"factors":4,"iterations":10,"reg":10,"seed":42}`

| Rank | Programme id | Title | Category | Distinct A viewers | Watched in A? | Relevant in B? |
|---:|---:|---|---|---:|:---:|:---:|
| 1 | 90077612 | Х/ф 'Джек Райан: Теория хаоса'. | Фильмы | 570 | no | no |
| 2 | 90073874 | Х/ф 'Ковбои против пришельцев'. | Фильмы | 507 | no | no |
| 3 | 90081370 | Х/ф 'Шанхайский перевозчик'. | Фильмы | 389 | no | no |
| 4 | 90069608 | Х/ф 'На крючке'. | Фильмы | 434 | no | no |
| 5 | 90053932 | Х/ф 'Одиночка'. | Фильмы | 555 | no | no |

### PPR: AP@5 = 0.066666667

Configuration: `{"alpha":0.8,"batch_size":128,"max_iterations":500,"tolerance":1e-06}`

| Rank | Programme id | Title | Category | Distinct A viewers | Watched in A? | Relevant in B? |
|---:|---:|---|---|---:|:---:|:---:|
| 1 | 90053098 | Х/ф 'Мужество в бою'. | Фильмы | 221 | yes | no |
| 2 | 2400480 | ТСН. Неделя. | Инфо | 4,362 | no | no |
| 3 | 240081 | Факты. День. | Инфо | 4,062 | no | yes |
| 4 | 6500479 | 'Мир наизнанку'. | Познавательное | 3,610 | no | no |
| 5 | 2400508 | Сегодня. Итоги с Олегом Панютой. | Инфо | 3,608 | no | no |

### CBF: AP@5 = 0.000000000

Configuration: `{"idf":"smooth_A_only","profile_weight":"log1p(seconds/60)","tf":"binary","year":"first_year_decade"}`

| Rank | Programme id | Title | Category | Distinct A viewers | Watched in A? | Relevant in B? |
|---:|---:|---|---|---:|:---:|:---:|
| 1 | 90053098 | Х/ф 'Мужество в бою'. | Фильмы | 221 | yes | no |
| 2 | 90026021 | Х/ф 'Осада'. | Фильмы | 0 | no | no |
| 3 | 90041855 | Х/ф 'Доблесть'. | Фильмы | 14 | no | no |
| 4 | 90052034 | Х/ф 'Большое дело'. | Фильмы | 0 | no | no |
| 5 | 90042923 | Х/ф 'Город ангелов'. | Фильмы | 0 | no | no |

### association: AP@5 = 0.200000000

Configuration: `{"lift_strictly_greater_than":1,"minimum_support":20,"score":"max_confidence"}`

| Rank | Programme id | Title | Category | Distinct A viewers | Watched in A? | Relevant in B? |
|---:|---:|---|---|---:|:---:|:---:|
| 1 | 240081 | Факты. День. | Инфо | 4,062 | no | yes |
| 2 | 2400480 | ТСН. Неделя. | Инфо | 4,362 | no | no |
| 3 | 6500479 | 'Мир наизнанку'. | Познавательное | 3,610 | no | no |
| 4 | 2400508 | Сегодня. Итоги с Олегом Панютой. | Инфо | 3,608 | no | no |
| 5 | 2400467 | 'Викна-Новости'. | Инфо | 3,538 | no | no |

### hybrid: AP@5 = 0.000000000

Configuration: `{"MF":2.0,"PPR":2.0,"popularity":0.0,"rewatch":0.5}`

| Rank | Programme id | Title | Category | Distinct A viewers | Watched in A? | Relevant in B? |
|---:|---:|---|---|---:|:---:|:---:|
| 1 | 90053098 | Х/ф 'Мужество в бою'. | Фильмы | 221 | yes | no |
| 2 | 90077612 | Х/ф 'Джек Райан: Теория хаоса'. | Фильмы | 570 | no | no |
| 3 | 90073874 | Х/ф 'Ковбои против пришельцев'. | Фильмы | 507 | no | no |
| 4 | 90081370 | Х/ф 'Шанхайский перевозчик'. | Фильмы | 389 | no | no |
| 5 | 90069608 | Х/ф 'На крючке'. | Фильмы | 434 | no | no |
