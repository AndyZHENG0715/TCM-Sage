# Seed crosswalk review preview

Status: pending human review (do not ingest as L0 until approved)

Total candidates: 37
- Symptom: 30
- Herb: 7

## Symptom candidates

| candidate_id | neijing_name | symmap_name | neijing_frequency |
|---|---|---|---:|
| symptom::汗出 | 汗出 | 汗出 | 1 |
| symptom::肿胀 | 肿胀 | 肿胀 | 1 |
| symptom::湿痹 | 湿痹 | 湿痹 | 1 |
| symptom::口苦 | 口苦 | 口苦 | 1 |
| symptom::寒热 | 寒热 | 寒热 | 1 |
| symptom::腹胀 | 腹胀 | 腹胀 | 1 |
| symptom::腰痛 | 腰痛 | 腰痛 | 1 |
| symptom::心痛 | 心痛 | 心痛 | 1 |
| symptom::头痛 | 头痛 | 头痛 | 1 |
| symptom::齿痛 | 齿痛 | 齿痛 | 1 |
| symptom::痈疽 | 痈疽 | 痈疽 | 1 |
| symptom::癫狂 | 癫狂 | 癫狂 | 1 |
| symptom::短气 | 短气 | 短气 | 1 |
| symptom::少气 | 少气 | 少气 | 1 |
| symptom::喘 | 喘 | 喘 | 1 |
| symptom::口干 | 口干 | 口干 | 1 |
| symptom::耳聋 | 耳聋 | 耳聋 | 1 |
| symptom::腹满 | 腹满 | 腹满 | 1 |
| symptom::下血 | 下血 | 下血 | 1 |
| symptom::耳鸣 | 耳鸣 | 耳鸣 | 1 |
| symptom::腹中胀满 | 腹中胀满 | 腹中胀满 | 1 |
| symptom::喉痹 | 喉痹 | 喉痹 | 1 |
| symptom::小便不利 | 小便不利 | 小便不利 | 1 |
| symptom::大便不利 | 大便不利 | 大便不利 | 1 |
| symptom::胀满 | 胀满 | 胀满 | 1 |
| symptom::腹痛 | 腹痛 | 腹痛 | 1 |
| symptom::呃逆 | 呃逆 | 呃逆 | 1 |
| symptom::身痛 | 身痛 | 身痛 | 1 |
| symptom::积聚 | 积聚 | 积聚 | 1 |
| symptom::发热 | 发热 | 发热 | 1 |

## Herb candidates

| candidate_id | neijing_name | symmap_name | neijing_frequency |
|---|---|---|---:|
| herb::竹茹 | 竹茹 | 竹茹 | 1 |
| herb::人参 | 人参 | 人参 | 1 |
| herb::甘草 | 甘草 | 甘草 | 1 |
| herb::生姜 | 生姜 | 生姜 | 1 |
| herb::大枣 | 大枣 | 大枣 | 1 |
| herb::金银花 | 金银花 | 金银花 | 1 |
| herb::王不留行 | 王不留行 | 王不留行 | 1 |

## Human review decisions (domain expert informed)

- Approved: 33
- Rejected: 4

Rejected candidates and rationale:
- `symptom::心痛`: requires source context (`心中痛` vs `心下痛`) before alignment.
- `symptom::寒热`: treated as pattern/ba-gang framework concept, not a standalone symptom node.
- `symptom::积聚`: disease-name axis mismatch; should be handled on Disease/Pattern typing path.
- `symptom::癫狂`: clinical usage distinguishes 癫 and 狂; avoid merged direct mapping.
