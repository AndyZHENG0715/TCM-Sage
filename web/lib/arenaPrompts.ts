export const ARENA_SAMPLE_PROMPTS = [
    { label: "麻黄汤组成", query: "麻黄汤的完整药物组成是什么？请列出所有药材及其剂量比例。" },
    { label: "十八反·乌头禁忌", query: "乌头反哪些药物？请列出十八反中乌头类的配伍禁忌。" },
    { label: "太阳病提纲", query: "《伤寒论》中太阳病的提纲条文是什么？请引述原文并解释。" },
    { label: "小柴胡汤方义", query: "小柴胡汤中“和解少阳”的具体方义是什么？各药物在方中扮演什么角色？" },
    { label: "四逆汤组成", query: "四逆汤的药物组成及煎服法是什么？请引用原文。" },
    { label: "桂枝汤服法", query: "桂枝汤服药后的护理要点是什么？包括啰粥、温覆、禁忌。" },
    { label: "促结代脉鉴别", query: "什么是促脉？与结脉、代脉的区别是什么？" },
    { label: "补中益气汤考证", query: "补中益气汤首见于哪本著作？作者是谁？组成是什么？" },
    { label: "刘寄奴药材", query: "刘寄奴是什么药材？其性味归经和功效是什么？" },
    { label: "桂枝汤类方鉴别", query: "桂枝汤、桂枝加葛根汤、桂枝加厚朴杏子汤的区别是什么？" },
] as const satisfies readonly { label: string; query: string }[];

export const ARENA_MODEL_PRESETS = [
    { label: "Flash", value: "qwen-turbo", description: "轻量快速" },
    { label: "Plus", value: "qwen-plus", description: "均衡性价比" },
    { label: "Max", value: "qwen-max", description: "旗舰性能" },
] as const satisfies readonly {
    label: string;
    value: string;
    description: string;
}[];
