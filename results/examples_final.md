# Final validation examples

All examples use fit window A, B truth TK, the same 6,376 candidates, and the frozen selected configurations. The reranker replays its internal-A validation fit. Saved aggregate metrics are unchanged; no searches were rerun.
**Selection rules:** heaviest = test user with the most A airings watched for at least 300 seconds; light = test user with the fewest A airings completed at least 80%, among users with at least 10 such A airings, at least one >=300-second A airing, and nonempty B TK. Both ties use ascending user_id. The earlier zero-history user is retained separately as the cold-start case.
**Definitions:** watched-in-A means at least one >=300-second A airing (the shared history rule). Completion counts use >=80% of full airing duration without a 300-second minimum. Titles/categories are display-only schedule lookups, preferring the latest A record and using B when an id has no A metadata.
**Fill provenance:** **†** marks an entry appended by the shared popularity completion, including forced cold starts. An item is not marked merely because it also occurs in the popularity list. Provenance was traced from each method's pre-fill proposals and checked against `ValidationProtocol.finalize_predictions`.

## Heaviest viewer: `13571199046623737884`

A history: **2,022 >=300s airings**, **649 >=300s programmes**, **9,388,931 qualifying watched seconds**; **1,014 >=80% airings** across **256 completed programmes**. Shared cold start: **no**.

### A top five by >=80% airing count

| Programme id | Title | Category | A completed airings |
|---:|---|---|---:|
| 2400480 | ТСН. Неделя. | Инфо | 73 |
| 240081 | Факты. День. | Инфо | 72 |
| 12001682 | М/с 'Губка Боб Квадратные Штаны', 92 с. 'Губка Боб Атлантические Штаны'. | Детям | 67 |
| 2400508 | Сегодня. Итоги с Олегом Панютой. | Инфо | 42 |
| 20088 | 'Завтрак с 1+1'. Информационно-развлекательная программа. | Развлечения | 40 |

### B TK truth

| Programme id | Title | Category | B completed airings |
|---:|---|---|---:|
| 240081 | Факты. День. | Инфо | 10 |
| 12001734 | М/с 'Волшебные покровители', 92 с. 'Чувства Чина'. | Детям | 2 |
| 200352 | 'Вечерний квартал', 17 сезон, 4 эп. | Развлечения | 1 |
| 700369 | 'Ліга сміху'. | Развлечения | 1 |
| 2400467 | 'Викна-Новости'. | Инфо | 1 |

### AP@5 and popularity completion

| Method | AP@5 under TK | Filled slots / 5 |
|---|---:|---:|
| popularity | 0.180000000 | 0 |
| rewatch_days | 0.200000000 | 0 |
| MF | 0.000000000 | 0 |
| PPR | 0.180000000 | 0 |
| CBF | 0.000000000 | 0 |
| association rules | 0.300000000 | 0 |
| slot | 0.066666667 | 0 |
| exposure | 0.200000000 | 0 |
| hybrid_v4 | 0.200000000 | 0 |
| rerank_v5 | 0.200000000 | 0 |

### Validation Top-5 lists

| Method | Rank | Programme id | Title | Category | Watched in A? | In TK? | Origin |
|---|---:|---:|---|---|:---:|:---:|---|
| popularity | 1 | 2400480 | ТСН. Неделя. | Инфо | yes | no | own score |
| popularity | 2 | 240081 | Факты. День. | Инфо | yes | yes | own score |
| popularity | 3 | 6500479 | 'Мир наизнанку'. | Познавательное | yes | no | own score |
| popularity | 4 | 2400508 | Сегодня. Итоги с Олегом Панютой. | Инфо | yes | no | own score |
| popularity | 5 | 2400467 | 'Викна-Новости'. | Инфо | yes | yes | own score |
| rewatch_days | 1 | 240081 | Факты. День. | Инфо | yes | yes | own score |
| rewatch_days | 2 | 2400480 | ТСН. Неделя. | Инфо | yes | no | own score |
| rewatch_days | 3 | 2400508 | Сегодня. Итоги с Олегом Панютой. | Инфо | yes | no | own score |
| rewatch_days | 4 | 6500479 | 'Мир наизнанку'. | Познавательное | yes | no | own score |
| rewatch_days | 5 | 400335 | 'Хата на тата', 6 сезон, 5 эп. | Развлечения | yes | no | own score |
| MF | 1 | 90077687 | Х/ф 'Великий уравнитель'. | Фильмы | yes | no | own score |
| MF | 2 | 2500430 | 'Затерянный мир'. | Познавательное | yes | no | own score |
| MF | 3 | 90076295 | Х/ф 'Призрачный патруль'. | Фильмы | yes | no | own score |
| MF | 4 | 90081867 | Х/ф 'Миссия невыполнима: Последствия'. | Фильмы | yes | no | own score |
| MF | 5 | 2500413 | Гражданская оборона. | Инфо | yes | no | own score |
| PPR | 1 | 2400480 | ТСН. Неделя. | Инфо | yes | no | own score |
| PPR | 2 | 240081 | Факты. День. | Инфо | yes | yes | own score |
| PPR | 3 | 6500479 | 'Мир наизнанку'. | Познавательное | yes | no | own score |
| PPR | 4 | 2400508 | Сегодня. Итоги с Олегом Панютой. | Инфо | yes | no | own score |
| PPR | 5 | 2400467 | 'Викна-Новости'. | Инфо | yes | yes | own score |
| CBF | 1 | 500346 | 'Право на власть'. | Развлечения | yes | no | own score |
| CBF | 2 | 500368 | Ток-шоу '60 минут' с Ольгой Скабеевой и Евгением Поповым. | Развлечения | no | no | own score |
| CBF | 3 | 400357 | 'Холостяк', 10 сезон, 11 эп. | Развлечения | yes | no | own score |
| CBF | 4 | 400373 | Премьера. 'Холостяк'. Реалити-шоу, 7 сезон, 12 эп. | Развлечения | no | no | own score |
| CBF | 5 | 500315 | Ток-шоу 'Касается каждого'. | Развлечения | yes | no | own score |
| association rules | 1 | 240081 | Факты. День. | Инфо | yes | yes | own score |
| association rules | 2 | 2400480 | ТСН. Неделя. | Инфо | yes | no | own score |
| association rules | 3 | 2400508 | Сегодня. Итоги с Олегом Панютой. | Инфо | yes | no | own score |
| association rules | 4 | 2400467 | 'Викна-Новости'. | Инфо | yes | yes | own score |
| association rules | 5 | 6500479 | 'Мир наизнанку'. | Познавательное | yes | no | own score |
| slot | 1 | 2400480 | ТСН. Неделя. | Инфо | yes | no | own score |
| slot | 2 | 20088 | 'Завтрак с 1+1'. Информационно-развлекательная программа. | Развлечения | yes | no | own score |
| slot | 3 | 2400467 | 'Викна-Новости'. | Инфо | yes | yes | own score |
| slot | 4 | 2400508 | Сегодня. Итоги с Олегом Панютой. | Инфо | yes | no | own score |
| slot | 5 | 6500479 | 'Мир наизнанку'. | Познавательное | yes | no | own score |
| exposure | 1 | 240081 | Факты. День. | Инфо | yes | yes | own score |
| exposure | 2 | 2400480 | ТСН. Неделя. | Инфо | yes | no | own score |
| exposure | 3 | 2400508 | Сегодня. Итоги с Олегом Панютой. | Инфо | yes | no | own score |
| exposure | 4 | 6500479 | 'Мир наизнанку'. | Познавательное | yes | no | own score |
| exposure | 5 | 12001682 | М/с 'Губка Боб Квадратные Штаны', 92 с. 'Губка Боб Атлантические Штаны'. | Детям | yes | no | own score |
| hybrid_v4 | 1 | 240081 | Факты. День. | Инфо | yes | yes | own score |
| hybrid_v4 | 2 | 12001682 | М/с 'Губка Боб Квадратные Штаны', 92 с. 'Губка Боб Атлантические Штаны'. | Детям | yes | no | own score |
| hybrid_v4 | 3 | 2400480 | ТСН. Неделя. | Инфо | yes | no | own score |
| hybrid_v4 | 4 | 2400508 | Сегодня. Итоги с Олегом Панютой. | Инфо | yes | no | own score |
| hybrid_v4 | 5 | 20088 | 'Завтрак с 1+1'. Информационно-развлекательная программа. | Развлечения | yes | no | own score |
| rerank_v5 | 1 | 240081 | Факты. День. | Инфо | yes | yes | own score |
| rerank_v5 | 2 | 2400480 | ТСН. Неделя. | Инфо | yes | no | own score |
| rerank_v5 | 3 | 2400508 | Сегодня. Итоги с Олегом Панютой. | Инфо | yes | no | own score |
| rerank_v5 | 4 | 12001682 | М/с 'Губка Боб Квадратные Штаны', 92 с. 'Губка Боб Атлантические Штаны'. | Детям | yes | no | own score |
| rerank_v5 | 5 | 200337 | Утро с Интером. | Развлечения | yes | no | own score |

## Light viewer: `5923732481008255736`

A history: **116 >=300s airings**, **67 >=300s programmes**, **186,049 qualifying watched seconds**; **10 >=80% airings** across **8 completed programmes**. Shared cold start: **no**.

### A top five by >=80% airing count

| Programme id | Title | Category | A completed airings |
|---:|---|---|---:|
| 240081 | Факты. День. | Инфо | 2 |
| 2400508 | Сегодня. Итоги с Олегом Панютой. | Инфо | 2 |
| 200344 | 'ЧистоNews'. | Развлечения | 1 |
| 200432 | Дизель-шоу. | Развлечения | 1 |
| 700369 | 'Ліга сміху'. | Развлечения | 1 |

### B TK truth

| Programme id | Title | Category | B completed airings |
|---:|---|---|---:|
| 2400508 | Сегодня. Итоги с Олегом Панютой. | Инфо | 4 |
| 10003040 | Т/с 'Короткое слово 'нет', 12 с. | Сериалы | 4 |
| 500281 | Ток-шоу 'Говорит Украина'. | Развлечения | 3 |
| 240081 | Факты. День. | Инфо | 2 |
| 500315 | Ток-шоу 'Касается каждого'. | Развлечения | 2 |

### AP@5 and popularity completion

| Method | AP@5 under TK | Filled slots / 5 |
|---|---:|---:|
| popularity | 0.200000000 | 0 |
| rewatch_days | 0.353333333 | 0 |
| MF | 0.200000000 | 0 |
| PPR | 0.200000000 | 0 |
| CBF | 0.000000000 | 0 |
| association rules | 0.333333333 | 0 |
| slot | 0.383333333 | 0 |
| exposure | 0.483333333 | 0 |
| hybrid_v4 | 0.400000000 | 0 |
| rerank_v5 | 0.453333333 | 0 |

### Validation Top-5 lists

| Method | Rank | Programme id | Title | Category | Watched in A? | In TK? | Origin |
|---|---:|---:|---|---|:---:|:---:|---|
| popularity | 1 | 2400480 | ТСН. Неделя. | Инфо | yes | no | own score |
| popularity | 2 | 240081 | Факты. День. | Инфо | yes | yes | own score |
| popularity | 3 | 6500479 | 'Мир наизнанку'. | Познавательное | yes | no | own score |
| popularity | 4 | 2400508 | Сегодня. Итоги с Олегом Панютой. | Инфо | yes | yes | own score |
| popularity | 5 | 2400467 | 'Викна-Новости'. | Инфо | yes | no | own score |
| rewatch_days | 1 | 2400480 | ТСН. Неделя. | Инфо | yes | no | own score |
| rewatch_days | 2 | 240081 | Факты. День. | Инфо | yes | yes | own score |
| rewatch_days | 3 | 500281 | Ток-шоу 'Говорит Украина'. | Развлечения | yes | yes | own score |
| rewatch_days | 4 | 6500479 | 'Мир наизнанку'. | Познавательное | yes | no | own score |
| rewatch_days | 5 | 2400508 | Сегодня. Итоги с Олегом Панютой. | Инфо | yes | yes | own score |
| MF | 1 | 2400480 | ТСН. Неделя. | Инфо | yes | no | own score |
| MF | 2 | 2400508 | Сегодня. Итоги с Олегом Панютой. | Инфо | yes | yes | own score |
| MF | 3 | 2400467 | 'Викна-Новости'. | Инфо | yes | no | own score |
| MF | 4 | 240081 | Факты. День. | Инфо | yes | yes | own score |
| MF | 5 | 6500479 | 'Мир наизнанку'. | Познавательное | yes | no | own score |
| PPR | 1 | 2400480 | ТСН. Неделя. | Инфо | yes | no | own score |
| PPR | 2 | 240081 | Факты. День. | Инфо | yes | yes | own score |
| PPR | 3 | 6500479 | 'Мир наизнанку'. | Познавательное | yes | no | own score |
| PPR | 4 | 2400508 | Сегодня. Итоги с Олегом Панютой. | Инфо | yes | yes | own score |
| PPR | 5 | 200352 | 'Вечерний квартал', 17 сезон, 4 эп. | Развлечения | yes | no | own score |
| CBF | 1 | 700389 | Кто сверху?, 8 сезон, 14 эп. | Развлечения | yes | no | own score |
| CBF | 2 | 500346 | 'Право на власть'. | Развлечения | no | no | own score |
| CBF | 3 | 500368 | Ток-шоу '60 минут' с Ольгой Скабеевой и Евгением Поповым. | Развлечения | no | no | own score |
| CBF | 4 | 200497 | 'Засекреченные списки'. Документальный спецпроект. | Развлечения | no | no | own score |
| CBF | 5 | 500314 | 'Вечерний Ургант'. Настя Ивлеева, Максим Фадеев, Валерия и Иосиф Пригожин. | Развлечения | no | no | own score |
| association rules | 1 | 240081 | Факты. День. | Инфо | yes | yes | own score |
| association rules | 2 | 2400480 | ТСН. Неделя. | Инфо | yes | no | own score |
| association rules | 3 | 2400508 | Сегодня. Итоги с Олегом Панютой. | Инфо | yes | yes | own score |
| association rules | 4 | 6500479 | 'Мир наизнанку'. | Познавательное | yes | no | own score |
| association rules | 5 | 2400467 | 'Викна-Новости'. | Инфо | yes | no | own score |
| slot | 1 | 2400480 | ТСН. Неделя. | Инфо | yes | no | own score |
| slot | 2 | 240081 | Факты. День. | Инфо | yes | yes | own score |
| slot | 3 | 500281 | Ток-шоу 'Говорит Украина'. | Развлечения | yes | yes | own score |
| slot | 4 | 2400508 | Сегодня. Итоги с Олегом Панютой. | Инфо | yes | yes | own score |
| slot | 5 | 6500479 | 'Мир наизнанку'. | Познавательное | yes | no | own score |
| exposure | 1 | 240081 | Факты. День. | Инфо | yes | yes | own score |
| exposure | 2 | 2400480 | ТСН. Неделя. | Инфо | yes | no | own score |
| exposure | 3 | 500281 | Ток-шоу 'Говорит Украина'. | Развлечения | yes | yes | own score |
| exposure | 4 | 2400508 | Сегодня. Итоги с Олегом Панютой. | Инфо | yes | yes | own score |
| exposure | 5 | 6500479 | 'Мир наизнанку'. | Познавательное | yes | no | own score |
| hybrid_v4 | 1 | 240081 | Факты. День. | Инфо | yes | yes | own score |
| hybrid_v4 | 2 | 2400508 | Сегодня. Итоги с Олегом Панютой. | Инфо | yes | yes | own score |
| hybrid_v4 | 3 | 2400480 | ТСН. Неделя. | Инфо | yes | no | own score |
| hybrid_v4 | 4 | 2400467 | 'Викна-Новости'. | Инфо | yes | no | own score |
| hybrid_v4 | 5 | 200432 | Дизель-шоу. | Развлечения | yes | no | own score |
| rerank_v5 | 1 | 240081 | Факты. День. | Инфо | yes | yes | own score |
| rerank_v5 | 2 | 2400480 | ТСН. Неделя. | Инфо | yes | no | own score |
| rerank_v5 | 3 | 2400508 | Сегодня. Итоги с Олегом Панютой. | Инфо | yes | yes | own score |
| rerank_v5 | 4 | 2400467 | 'Викна-Новости'. | Инфо | yes | no | own score |
| rerank_v5 | 5 | 500281 | Ток-шоу 'Говорит Украина'. | Развлечения | yes | yes | own score |

## Cold-start case: `319675768840678676`

This user has zero qualifying >=300-second A airings and zero >=80% A completions. Every method listed above returns the same shared popularity fallback list; all five slots are popularity-filled.

### B TK truth

| Programme id | Title | Category | B completed airings |
|---:|---|---|---:|
| 1000589 | Т/с 'Отчаянные домохозяйки', 3 сезон, 19 с. 'Боже, как хорошо'. | Сериалы | 2 |
| 10002454 | Т/с 'Гримм', 1 сезон, 12 с. | Сериалы | 1 |
| 90074341 | Х/ф 'Белоснежка: Месть гномов'. | Фильмы | 1 |

### Shared fallback Top-5 (every method)

| Rank | Programme id | Title | Category | In TK? | Origin |
|---:|---:|---|---|:---:|---|
| 1 | 2400480† | ТСН. Неделя. | Инфо | no | popularity fill: cold start |
| 2 | 240081† | Факты. День. | Инфо | no | popularity fill: cold start |
| 3 | 6500479† | 'Мир наизнанку'. | Познавательное | no | popularity fill: cold start |
| 4 | 2400508† | Сегодня. Итоги с Олегом Панютой. | Инфо | no | popularity fill: cold start |
| 5 | 2400467† | 'Викна-Новости'. | Инфо | no | popularity fill: cold start |

All methods: AP@5 under TK = **0.000000000**; popularity-filled slots = **5/5**.
