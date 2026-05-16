export const ARENA_SAMPLE_PROMPTS = [
    // Proven RAG advantage: LLM hallucinates on these, RAG retrieves correct source
    { label: "水蛭性味功效", query: "水蛭的性味是什么？出自哪本经典？主治什么？" },
    { label: "蛇床子配伍禁忌", query: "蛇床子有哪些配伍禁忌？恶什么药？" },
    { label: "麻黄汤组成剂量", query: "麻黄汤的完整药物组成和剂量是什么？请引用原文。" },
    { label: "独活寄生汤全方", query: "独活寄生汤有多少味药？请全部列出并注明剂量。" },
    { label: "附子性味原文", query: "附子的性味，神农本草经原文怎么说？" },
    // Improved retrieval after clause-level re-ingestion
    { label: "小柴胡汤组成", query: "小柴胡汤的组成是什么？请列出全部药味和剂量。" },
    { label: "马兑铃本草记载", query: "马兑铃在本草纲目中的记载是什么？有什么别名？" },
    { label: "丹溪痰证治法", query: "《丹溪心法》中关于痰证的治法有哪些？分别用什么药？" },
    { label: "太阳病提纲", query: "《伤寒论》中太阳病的提纲条文是什么？请引述原文并解释。" },
    { label: "十八反·乌头禁忌", query: "乌头反哪些药物？请列出十八反中乌头类的配伍禁忌。" },
] as const satisfies readonly { label: string; query: string }[];

export const ARENA_MODEL_PRESETS = [
    { label: "Flash", value: "qwen-flash", description: "轻量快速" },
    { label: "Plus", value: "qwen-plus", description: "均衡性价比" },
    { label: "Max", value: "qwen-max", description: "旗舰性能" },
] as const satisfies readonly {
    label: string;
    value: string;
    description: string;
}[];
