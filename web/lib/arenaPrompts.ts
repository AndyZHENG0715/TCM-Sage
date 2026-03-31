export const ARENA_SAMPLE_PROMPTS = [
    "什么是气血？",
    "阴阳的基本概念是什么？",
    "五行理论如何解释人体脏腑功能？",
    "经络系统的主要功能是什么？",
    "头痛应该用什么方剂？",
    "失眠的中医治疗方法有哪些？",
    "感冒风寒证的治疗原则是什么？",
    "患者男，42岁，反复口苦口干2周，胸胁胀满，急躁易怒，舌红苔黄，脉弦数，请先判断可能的证型，再说明辨证依据与常见治法。",
    "患者女，35岁，近3个月入睡困难、多梦易醒，伴心悸健忘、疲倦乏力、面色少华，舌淡苔薄白，脉细弱；请给出辨证思路、治则和代表方的选择理由。",
    "患者男，58岁，平素体形偏胖，近半年胸闷痰多、食后脘痞、肢体困重、口黏不渴，舌苔厚腻，脉滑；如果从中医角度分析，应如何归纳病机并分层讨论治疗策略？",
] as const satisfies readonly string[];

export const ARENA_MODEL_PRESETS = [
    { label: "Flash", value: "qwen-turbo", description: "轻量快速" },
    { label: "Plus", value: "qwen-plus", description: "均衡性价比" },
    { label: "Max", value: "qwen-max", description: "旗舰性能" },
] as const satisfies readonly {
    label: string;
    value: string;
    description: string;
}[];
