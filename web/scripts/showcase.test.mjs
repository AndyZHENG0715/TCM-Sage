import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import vm from "node:vm";
import ts from "typescript";

const web = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const root = resolve(web, "..");
const readJson = path => JSON.parse(readFileSync(path, "utf8"));
const fixtures = readJson(resolve(web, "lib/showcase-data.json"));
const chunks = readJson(resolve(root, "data/processed/chunks.json"));
const graph = readJson(resolve(root, "data/graph/symmap/symmap_entities.json"));

function loadShowcase() {
    const runtimeModule = { exports: {} };
    const source = ts.transpileModule(readFileSync(resolve(web, "lib/showcase.ts"), "utf8"), {
        compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
    }).outputText;
    const context = vm.createContext({
        module: runtimeModule, exports: runtimeModule.exports, process: { env: {} },
        setTimeout: callback => { callback(); return 1; },
        require: path => {
            assert.equal(path, "./showcase-data.json");
            return fixtures;
        },
        fetch: () => { throw new Error("Replay attempted network access"); },
    });
    vm.runInContext(source, context);
    return runtimeModule.exports;
}

test("all six recorded sources remain exact corpus excerpts with stable fingerprints", () => {
    assert.equal(fixtures.examples.length, 6);
    assert.equal(new Set(fixtures.examples.map(e => e.id)).size, 6);
    const byId = new Map(chunks.map(c => [c.id, c]));
    for (const example of fixtures.examples) {
        for (const citation of example.citations.filter(c => c.type === "text")) {
            assert.equal(citation.content, byId.get(citation.chunk_id)?.content);
            assert.equal(fixtures.contexts[citation.chunk_id].paragraph_text, citation.content);
            assert.equal(fixtures.contexts[citation.chunk_id].recorded_excerpt, true);
        }
    }
    assert.equal(fixtures.corpusSha256, createHash("sha256").update(readFileSync(resolve(root, "data/processed/chunks.json"))).digest("hex"));
    assert.equal(fixtures.graphSha256, createHash("sha256").update(readFileSync(resolve(root, "data/graph/symmap/symmap_entities.json"))).digest("hex"));
});

test("recorded graph relationships and provenance exist in the original graph", () => {
    const nodeIds = new Set(graph.entities.map(e => e.id));
    const relationships = new Set(graph.relationships.map(r => `${r.source}|${r.target}|${r.type}|${r.source_ref}`));
    for (const example of fixtures.examples) {
        for (const node of example.graph.nodes) assert.ok(nodeIds.has(node.id));
        for (const relation of fixtures.graphProvenance[example.graphEntity]) {
            assert.ok(relationships.has(`${relation.source}|${relation.target}|${relation.type}|${relation.source_ref}`));
        }
    }
});

test("English and Chinese replay finish from local data and honour cancellation", async () => {
    const app = loadShowcase();
    const example = app.RECORDED_EXAMPLES[0];
    for (const chinese of [false, true]) {
        let content = "";
        for await (const chunk of app.replayExample(example, chinese)) content += chunk;
        assert.equal(content, chinese ? example.answerZh : example.answer);
        assert.equal(app.recordedMessage(example, chinese).recorded, true);
    }
    const output = [];
    for await (const chunk of app.replayExample(example, false, { aborted: true })) output.push(chunk);
    assert.equal(output.length, 0);
});

test("source, book and graph navigation resolve locally; access code is memory-only", () => {
    const app = loadShowcase();
    const example = app.RECORDED_EXAMPLES[0];
    const citation = example.citations.find(c => c.type === "text");
    assert.equal(app.recordedContext(citation.chunk_id).paragraph_text, citation.content);
    assert.ok(app.recordedBook(app.recordedContext(citation.chunk_id).book).content.includes("桂枝"));
    assert.equal(app.recordedGraph(encodeURIComponent(example.graphEntity)).cited_ids[0], example.graph.cited_ids[0]);
    assert.ok(app.recordedGraphSearch(example.graphEntity, 20).length > 0);
    for (const snapshot of Object.values(fixtures.graphs)) {
        for (const node of snapshot.nodes) assert.ok(app.recordedGraph(node.id));
    }
    assert.equal(app.PRESENTATION_MODE, false);
    assert.equal(app.getLiveAccessCode(), "");
    app.setLiveAccessCode(" sample-project-code ");
    assert.equal(app.getLiveAccessCode(), "sample-project-code");
    assert.equal(loadShowcase().getLiveAccessCode(), "");
});
