"use strict";
const $ = (id) => document.getElementById(id);
let latestResult = null;
let methods = [];
const verdicts = {supported: "Được hỗ trợ", contradicted: "Có dấu hiệu mâu thuẫn", unsupported: "Chưa được hỗ trợ", unknown: "Chưa đủ thông tin"};
const node = (tag, text, className) => {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
};
function toggle(id, visible) { $(id).classList.toggle("hidden", !visible); }
function clearResult() {
  latestResult = null;
  toggle("result", false); toggle("error", false); toggle("empty-state", true);
}
function updateCount() { $("claim-count").textContent = `${$("claim").value.length.toLocaleString("vi-VN")} / 6.000`; }
function methodHelp() {
  const selected = methods.find((method) => method.id === $("method").value);
  if (selected) $("method-help").textContent = `${selected.requirement}. ${selected.configured ? "Đã có cấu hình; model được kiểm tra khi chạy." : "Chưa cấu hình; xem hướng dẫn README để bật."}`;
}
function renderResult(data) {
  latestResult = data;
  toggle("empty-state", false); toggle("result", true);
  const unknown = data.is_hallucination === null;
  const bad = data.is_hallucination === true;
  $("verdict-label").className = `verdict-tag ${unknown ? "unknown" : bad ? "bad" : ""}`;
  $("verdict-label").textContent = unknown ? "CHƯA ĐỦ BẰNG CHỨNG" : bad ? "CẦN KIỂM TRA LẠI" : "DƯỚI NGƯỠNG CẢNH BÁO";
  $("result-title").textContent = unknown ? "Cần thêm thông tin để kết luận." : bad ? "Có thông tin cần được xác minh." : "Chưa vượt ngưỡng cảnh báo.";
  $("result-explanation").textContent = data.explanation;
  $("latency").textContent = `${data.latency_ms.toLocaleString("vi-VN")} ms`;
  $("score-value").textContent = data.score === null ? "—" : data.score.toFixed(2);
  $("score-fill").style.width = `${(data.score ?? 0) * 100}%`;
  $("score-fill").style.backgroundColor = bad ? "#c08b68" : "#8eaa65";
  $("threshold-marker").style.left = `${data.threshold * 100}%`;
  $("threshold-note").textContent = `Ngưỡng: ${data.threshold}`;
  $("warnings").replaceChildren(...data.warnings.map((warning) => node("p", warning)));
  $("findings-count").textContent = `${data.findings.length} mệnh đề`;
  $("findings").replaceChildren(...data.findings.map((finding, i) => {
    const card = node("article", undefined, "finding");
    const head = node("div", undefined, "finding-head");
    head.append(node("strong", `${String(i + 1).padStart(2, "0")} · ${finding.claim}`), node("span", verdicts[finding.verdict]));
    card.append(head, node("p", finding.explanation));
    for (const evidence of finding.evidence) {
      const quote = node("blockquote");
      quote.append(node("small", `${evidence.source} · ${evidence.relation}`), node("span", evidence.quote));
      card.append(quote);
    }
    return card;
  }));
  toggle("components-wrapper", data.components.length > 0);
  $("components").replaceChildren(...data.components.map((component) => {
    const row = node("div", undefined, "component");
    row.append(node("span", component.method), node("strong", component.score === null ? "Chưa đủ thông tin" : component.score.toFixed(3)));
    return row;
  }));
}
$("claim").addEventListener("input", updateCount);
$("method").addEventListener("change", methodHelp);
$("clear-form").addEventListener("click", () => {
  $("detect-form").reset(); clearResult(); updateCount(); methodHelp(); $("claim").focus();
});
$("detect-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!$("claim").value.trim()) { $("claim").focus(); return; }
  const payload = {
    claim: $("claim").value.trim(), context: $("context").value.trim(), question: $("question").value.trim(),
    samples: $("samples").value.split(/\n\s*\n/).map((s) => s.trim()).filter(Boolean),
    method: $("method").value, threshold: Number($("threshold").value),
  };
  clearResult(); toggle("empty-state", false); toggle("loading", true);
  $("submit").disabled = true;
  try {
    const response = await fetch("/detect", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
    const data = await response.json();
    if (!response.ok) {
      const message = typeof data.detail === "string" ? data.detail : "Dữ liệu không hợp lệ. Kiểm tra độ dài, số mẫu và ngưỡng quyết định.";
      throw new Error(message);
    }
    renderResult(data);
  } catch (error) {
    $("error").textContent = error.message || "Không thể kết nối đến máy chủ.";
    toggle("error", true);
  } finally {
    toggle("loading", false); $("submit").disabled = false;
  }
});
$("download").addEventListener("click", () => {
  if (!latestResult) return;
  const blob = new Blob([JSON.stringify(latestResult, null, 2)], {type: "application/json"});
  const url = URL.createObjectURL(blob);
  const link = node("a"); link.href = url; link.download = "hallucination-report.json"; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
async function initialize() {
  try {
    const [methodResponse, exampleResponse] = await Promise.all([fetch("/methods"), fetch("/examples")]);
    if (!methodResponse.ok || !exampleResponse.ok) throw new Error("Không tải được cấu hình máy chủ.");
    methods = (await methodResponse.json()).methods;
    $("method").replaceChildren(...methods.map((method) => {
      const option = node("option", `${method.name}${method.configured ? "" : " · cần cấu hình"}`);
      option.value = method.id; return option;
    }));
    methodHelp();
    const examples = (await exampleResponse.json()).examples;
    $("examples").replaceChildren(...examples.map((example) => {
      const button = node("button", example.title); button.type = "button";
      button.append(node("span", "↗"));
      button.addEventListener("click", () => {
        $("claim").value = example.claim; $("context").value = example.context;
        $("question").value = example.question; $("samples").value = "";
        clearResult(); updateCount(); $("claim").focus();
      });
      return button;
    }));
  } catch (error) { $("method-help").textContent = error.message; }
}
initialize();
