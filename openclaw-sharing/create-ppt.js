const pptxgen = require("pptxgenjs");

// 配色方案 - Charcoal Minimal (现代科技感)
const COLORS = {
  primary: "1E2761",      // 深海军蓝
  secondary: "CADCFC",    // 冰蓝
  accent: "0891B2",       // 青色强调
  text: "1E293B",         // 深灰文字
  lightText: "FFFFFF",    // 白色文字
  background: "F8FAFC",   // 浅灰背景
  cardBg: "FFFFFF",       // 卡片背景
  muted: "64748B"         // 灰色辅助文字
};

let pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "Damon";
pres.title = "OpenClaw 分享 - AI Agent 工作流实践";

// ========== 第1页：标题页 ==========
let slide1 = pres.addSlide();
slide1.background = { color: COLORS.primary };

// 主标题
slide1.addText("OpenClaw 分享", {
  x: 0.5, y: 2.0, w: 9, h: 1.2,
  fontSize: 48, fontFace: "Arial", bold: true,
  color: COLORS.lightText, align: "center"
});

// 副标题
slide1.addText("AI Agent 工作流实践与 Harness Engineering", {
  x: 0.5, y: 3.3, w: 9, h: 0.6,
  fontSize: 22, fontFace: "Arial",
  color: COLORS.secondary, align: "center"
});

// 底部信息
slide1.addText("Damon | 2026", {
  x: 0.5, y: 5.0, w: 9, h: 0.4,
  fontSize: 14, fontFace: "Arial",
  color: COLORS.muted, align: "center"
});

// ========== 第2页：需求气泡图 ==========
let slide2 = pres.addSlide();
slide2.background = { color: COLORS.background };

// 标题
slide2.addText("我的 AI 使用需求全景", {
  x: 0.5, y: 0.3, w: 9, h: 0.6,
  fontSize: 32, fontFace: "Arial", bold: true,
  color: COLORS.text
});

// 中心气泡 - Damon
slide2.addShape(pres.shapes.OVAL, {
  x: 4.0, y: 2.2, w: 2.0, h: 1.2,
  fill: { color: COLORS.primary }
});
slide2.addText("Damon", {
  x: 4.0, y: 2.5, w: 2.0, h: 0.6,
  fontSize: 18, fontFace: "Arial", bold: true,
  color: COLORS.lightText, align: "center", valign: "middle"
});

// 需求气泡数据
const bubbles = [
  { label: "信息获取", x: 1.5, y: 1.2, color: COLORS.accent },
  { label: "知识沉淀", x: 6.5, y: 1.2, color: COLORS.accent },
  { label: "项目开发", x: 1.5, y: 3.5, color: COLORS.accent },
  { label: "日常协作", x: 6.5, y: 3.5, color: COLORS.accent },
  { label: "探索实验", x: 4.0, y: 4.3, color: COLORS.accent }
];

bubbles.forEach(b => {
  slide2.addShape(pres.shapes.OVAL, {
    x: b.x, y: b.y, w: 2.2, h: 1.0,
    fill: { color: b.color }, shadow: { type: "outer", blur: 4, offset: 2, color: "000000", opacity: 0.1 }
  });
  slide2.addText(b.label, {
    x: b.x, y: b.y + 0.25, w: 2.2, h: 0.5,
    fontSize: 14, fontFace: "Arial", bold: true,
    color: COLORS.lightText, align: "center", valign: "middle"
  });
});

// 说明文字
slide2.addText("这些需求不是独立的，是相互关联的", {
  x: 0.5, y: 5.0, w: 9, h: 0.4,
  fontSize: 14, fontFace: "Arial", italic: true,
  color: COLORS.muted, align: "center"
});

// ========== 第3页：需求→项目关联 ==========
let slide3 = pres.addSlide();
slide3.background = { color: COLORS.background };

// 标题
slide3.addText("需求如何落地为项目", {
  x: 0.5, y: 0.3, w: 9, h: 0.6,
  fontSize: 32, fontFace: "Arial", bold: true,
  color: COLORS.text
});

// 项目卡片
const projects = [
  { need: "快速研究陌生领域", project: "动物声音通信研究", method: "Hermes 搜索 → 整理 → 写入文档", level: "Level 1" },
  { need: "知识沉淀", project: "2nd_brain 知识库", method: "OpenClaw 长期维护 + cron", level: "Level 3" },
  { need: "写代码/调试", project: "ClawTeam 多 Agent", method: "Claude Code 开发 + OpenClaw 运行", level: "Level 2-3" },
  { need: "多 Agent 协作", project: "Nemo ↔ Outis A2A", method: "OpenClaw 插件 + 网络配置", level: "Level 3" }
];

// 表头
slide3.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 1.1, w: 9, h: 0.5,
  fill: { color: COLORS.primary }
});
slide3.addText([
  { text: "需求", options: { bold: true, color: COLORS.lightText, breakLine: false } },
], { x: 0.6, y: 1.15, w: 2, h: 0.4, fontSize: 12, fontFace: "Arial" });
slide3.addText("项目", { x: 2.6, y: 1.15, w: 2.2, h: 0.4, fontSize: 12, fontFace: "Arial", bold: true, color: COLORS.lightText });
slide3.addText("构建方式", { x: 4.8, y: 1.15, w: 3, h: 0.4, fontSize: 12, fontFace: "Arial", bold: true, color: COLORS.lightText });
slide3.addText("层次", { x: 8.0, y: 1.15, w: 1.4, h: 0.4, fontSize: 12, fontFace: "Arial", bold: true, color: COLORS.lightText });

// 数据行
projects.forEach((p, i) => {
  const y = 1.65 + i * 0.65;
  const bgColor = i % 2 === 0 ? "FFFFFF" : "F1F5F9";
  
  slide3.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: y, w: 9, h: 0.6,
    fill: { color: bgColor }, line: { color: "E2E8F0", width: 0.5 }
  });
  slide3.addText(p.need, { x: 0.6, y: y + 0.1, w: 2, h: 0.4, fontSize: 11, fontFace: "Arial", color: COLORS.text });
  slide3.addText(p.project, { x: 2.6, y: y + 0.1, w: 2.2, h: 0.4, fontSize: 11, fontFace: "Arial", color: COLORS.text });
  slide3.addText(p.method, { x: 4.8, y: y + 0.1, w: 3, h: 0.4, fontSize: 10, fontFace: "Arial", color: COLORS.muted });
  slide3.addText(p.level, { x: 8.0, y: y + 0.1, w: 1.4, h: 0.4, fontSize: 11, fontFace: "Arial", bold: true, color: COLORS.accent });
});

// 核心概念
slide3.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 4.4, w: 9, h: 0.9,
  fill: { color: COLORS.secondary }, shadow: { type: "outer", blur: 3, offset: 1, color: "000000", opacity: 0.08 }
});
slide3.addText("核心概念：让 AI 理解我的工作流，并嵌入其中", {
  x: 0.6, y: 4.55, w: 8.8, h: 0.6,
  fontSize: 16, fontFace: "Arial", bold: true,
  color: COLORS.primary, align: "center", valign: "middle"
});

// ========== 第4页：工具对比 ==========
let slide4 = pres.addSlide();
slide4.background = { color: COLORS.background };

// 标题
slide4.addText("三个 Agent 工具对比", {
  x: 0.5, y: 0.3, w: 9, h: 0.6,
  fontSize: 32, fontFace: "Arial", bold: true,
  color: COLORS.text
});

// 工具对比卡片
const tools = [
  { name: "OpenClaw", desc: "本地化定制 Agent 平台", color: COLORS.primary, features: ["Level 1-3 全支持", "多 Agent 协作", "自动化能力", "长期积累"] },
  { name: "Hermes Agent", desc: "云端通用 Agent", color: COLORS.accent, features: ["快速响应", "即用即走", "研究能力强", "低学习成本"] },
  { name: "Claude Code", desc: "代码专用 Agent", color: "0891B2", features: ["代码能力突出", "开发场景优化", "项目上下文", "快速迭代"] }
];

tools.forEach((tool, i) => {
  const x = 0.5 + i * 3.1;
  
  // 卡片背景
  slide4.addShape(pres.shapes.RECTANGLE, {
    x: x, y: 1.1, w: 2.9, h: 3.8,
    fill: { color: COLORS.cardBg }, shadow: { type: "outer", blur: 6, offset: 3, color: "000000", opacity: 0.1 }
  });
  
  // 顶部色条
  slide4.addShape(pres.shapes.RECTANGLE, {
    x: x, y: 1.1, w: 2.9, h: 0.15,
    fill: { color: tool.color }
  });
  
  // 工具名称
  slide4.addText(tool.name, {
    x: x + 0.1, y: 1.35, w: 2.7, h: 0.5,
    fontSize: 18, fontFace: "Arial", bold: true,
    color: tool.color, align: "center"
  });
  
  // 描述
  slide4.addText(tool.desc, {
    x: x + 0.1, y: 1.85, w: 2.7, h: 0.4,
    fontSize: 11, fontFace: "Arial",
    color: COLORS.muted, align: "center"
  });
  
  // 特性列表
  tool.features.forEach((f, fi) => {
    slide4.addText("✓ " + f, {
      x: x + 0.2, y: 2.4 + fi * 0.45, w: 2.5, h: 0.4,
      fontSize: 12, fontFace: "Arial",
      color: COLORS.text
    });
  });
});

// 底部总结
slide4.addText("不是「哪个更好」，而是「哪个更适合」", {
  x: 0.5, y: 5.0, w: 9, h: 0.4,
  fontSize: 14, fontFace: "Arial", italic: true,
  color: COLORS.muted, align: "center"
});

// ========== 第5页：Harness Engineering ==========
let slide5 = pres.addSlide();
slide5.background = { color: COLORS.background };

// 标题
slide5.addText("Harness Engineering：让 Agent 从工具变成伙伴", {
  x: 0.5, y: 0.3, w: 9, h: 0.6,
  fontSize: 28, fontFace: "Arial", bold: true,
  color: COLORS.text
});

// 三个层次
const levels = [
  { level: "Level 1", title: "即用即走", desc: "直接对话，完成任务\n无配置、无定制", example: "问 Hermes 一个问题" },
  { level: "Level 2", title: "配置适配", desc: "加载 Skills、配置 Memory\nAgent 理解你的偏好", example: "Hermes 记住你的工作习惯" },
  { level: "Level 3", title: "工作流嵌入", desc: "参与多阶段流程\n与其他 Agent 协作", example: "OpenClaw + 2nd_brain + cron" }
];

levels.forEach((l, i) => {
  const x = 0.5 + i * 3.1;
  
  // 卡片
  slide5.addShape(pres.shapes.RECTANGLE, {
    x: x, y: 1.1, w: 2.9, h: 2.8,
    fill: { color: COLORS.cardBg }, shadow: { type: "outer", blur: 4, offset: 2, color: "000000", opacity: 0.08 }
  });
  
  // 层次标签
  slide5.addShape(pres.shapes.RECTANGLE, {
    x: x, y: 1.1, w: 2.9, h: 0.5,
    fill: { color: COLORS.primary }
  });
  slide5.addText(l.level, {
    x: x, y: 1.15, w: 2.9, h: 0.4,
    fontSize: 16, fontFace: "Arial", bold: true,
    color: COLORS.lightText, align: "center", valign: "middle"
  });
  
  // 标题
  slide5.addText(l.title, {
    x: x + 0.1, y: 1.7, w: 2.7, h: 0.4,
    fontSize: 16, fontFace: "Arial", bold: true,
    color: COLORS.text, align: "center"
  });
  
  // 描述
  slide5.addText(l.desc, {
    x: x + 0.1, y: 2.15, w: 2.7, h: 0.8,
    fontSize: 11, fontFace: "Arial",
    color: COLORS.muted, align: "center"
  });
  
  // 示例
  slide5.addText("例：" + l.example, {
    x: x + 0.1, y: 3.0, w: 2.7, h: 0.4,
    fontSize: 10, fontFace: "Arial", italic: true,
    color: COLORS.accent, align: "center"
  });
});

// 箭头连接
slide5.addText("→", { x: 3.3, y: 2.3, w: 0.5, h: 0.5, fontSize: 24, color: COLORS.muted, align: "center" });
slide5.addText("→", { x: 6.4, y: 2.3, w: 0.5, h: 0.5, fontSize: 24, color: COLORS.muted, align: "center" });

// 底部提示
slide5.addShape(pres.shapes.RECTANGLE, {
  x: 0.5, y: 4.2, w: 9, h: 0.9,
  fill: { color: COLORS.secondary }
});
slide5.addText("不是所有项目都要 Level 3，不同需求适合不同层次", {
  x: 0.6, y: 4.35, w: 8.8, h: 0.6,
  fontSize: 15, fontFace: "Arial", bold: true,
  color: COLORS.primary, align: "center", valign: "middle"
});

// ========== 第6页：结语 ==========
let slide6 = pres.addSlide();
slide6.background = { color: COLORS.primary };

// 标题
slide6.addText("核心信息", {
  x: 0.5, y: 0.8, w: 9, h: 0.6,
  fontSize: 36, fontFace: "Arial", bold: true,
  color: COLORS.lightText, align: "center"
});

// 要点
const keyPoints = [
  "AI Agent 不是魔法，是工具",
  "不同需求需要不同的 Harness Level",
  "选择工具的关键：理解你的需求类型",
  "从 Level 1 开始，逐步深入",
  "让 Agent 从工具变成伙伴"
];

keyPoints.forEach((point, i) => {
  slide6.addText((i + 1) + ". " + point, {
    x: 1.5, y: 1.8 + i * 0.55, w: 7, h: 0.5,
    fontSize: 18, fontFace: "Arial",
    color: COLORS.secondary, align: "left"
  });
});

// 底部
slide6.addText("感谢聆听 | 欢迎交流", {
  x: 0.5, y: 4.8, w: 9, h: 0.5,
  fontSize: 16, fontFace: "Arial",
  color: COLORS.muted, align: "center"
});

// 保存文件
pres.writeFile({ fileName: "OpenClaw-Sharing.pptx" })
  .then(() => console.log("PPT created: OpenClaw-Sharing.pptx"))
  .catch(err => console.error("Error:", err));