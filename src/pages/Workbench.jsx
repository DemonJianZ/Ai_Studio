import React, { useState, useRef, useCallback, useEffect } from "react";
import {
  Upload,
  Image as ImageIcon,
  Wand2,
  Download,
  X,
  Play,
  Plus,
  Zap,
  MousePointer2,
  Layers,
  Loader2,
  Images,
  ImagePlus,
  Minus,
  Maximize,
  Trash2,
  Undo,
  Redo,
  Copy,
  Hand,
  ShoppingBag,
  CheckSquare,
  MessageSquare,
  AlertCircle,
  ChevronRight,
  ChevronDown,
  Info,
  Settings2,
  History,
  TrendingUp,
  RefreshCw,
  LayoutGrid,
  Square,
  CheckCircle2,
  Sun,
  Sparkles,
  Sliders,
  Palette,
  Clapperboard,
  Film,
  WifiOff,
  FileWarning,
  ArrowLeft,
  ArrowRight,
  Cpu,
  MoreHorizontal,
  RotateCcw,
  Link as LinkIcon,
  Server,
  Keyboard,
  Clipboard,
  Activity,
  Layout,
  Send,
  Bot,
} from "lucide-react";
import { useAuth } from "../auth/AuthProvider";

// ==========================================
// Config & Constants
// ==========================================
const generateId = () => Math.random().toString(36).substr(2, 9);
const GRID_SIZE = 20;
const MIN_ZOOM = 0.1;
const MAX_ZOOM = 3;
const CANVAS_KEY = "bananaflow_canvas_id";

const LOADING_TIPS = [
  "正在重塑光影氛围...",
  "AI 正在计算物体表面漫反射...",
  "正在生成帧间动态光流...",
  "正在计算物理碰撞与运动轨迹...",
  "正在渲染关键帧插值...",
  "正在构思光影布局...",
  "精彩马上呈现...",
];

const NODE_TYPES = {
  INPUT: "input",
  TEXT_INPUT: "text_input",
  PROCESSOR: "processor",
  POST_PROCESSOR: "post_processor",
  VIDEO_GEN: "video_gen",
  OUTPUT: "output",
};

const AI_MODELS = [
  { id: "gemini-3-pro-image-preview", name: "Gemini 3 Pro", vendor: "Google", icon: Sparkles },
  { id: "doubao-seedream-4.5", name: "Doubao 4.5", vendor: "ByteDance", icon: Zap },
];

const VIDEO_MODELS = [
  { id: "Doubao-Seedance-1.0-pro", name: "Doubao Seedance 1.0 Pro", vendor: "ByteDance" },
  { id: "Doubao-Seedance-1.5-pro", name: "Doubao Seedance 1.5 Pro", vendor: "ByteDance" },
];

const VIDEO_MODEL_1_0 = "Doubao-Seedance-1.0-pro";
const VIDEO_MODEL_1_5 = "Doubao-Seedance-1.5-pro";

const TOOL_CARDS = {
  bg_replace: {
    id: "bg_replace",
    name: "一键换背景",
    short: "换背景",
    icon: Layers,
    desc: "保留商品和手，仅替换背景",
    scenario: "主图合成 / 详情页",
    refLabel: "背景参考图",
    category: "generate",
    refRequired: true,
  },
  gesture_swap: {
    id: "gesture_swap",
    name: "参考图换手势",
    short: "换手势",
    icon: Hand,
    desc: "保留商品与背景，迁移手势",
    scenario: "模仿网红手势",
    refLabel: "手势参考图",
    category: "generate",
    refRequired: true,
  },
  product_swap: {
    id: "product_swap",
    name: "保留手势换商品",
    short: "换商品",
    icon: ShoppingBag,
    desc: "保留手势与背景，替换商品",
    scenario: "多商品复用模版",
    refLabel: "新商品图",
    category: "generate",
    refRequired: true,
  },
  text2img: {
    id: "text2img",
    name: "创意生成 (Text2Img)",
    short: "文生图",
    icon: Wand2,
    desc: "从零生成营销素材",
    scenario: "灵感构思",
    category: "generate",
    refRequired: false,
  },
  multi_image_generate: {
    id: "multi_image_generate",
    name: "图生图 (Img2Img)",
    short: "图生图",
    icon: ImageIcon,
    desc: "参考原图生成新图像",
    scenario: "风格迁移/重绘",
    category: "generate",
    refRequired: false,
  },
  relight: {
    id: "relight",
    name: "智能打光 (Relight)",
    short: "光影精修",
    icon: Sun,
    desc: '修复光线不自然，重塑光影',
    scenario: '解决"贴图感" / 氛围增强',
    refLabel: "光影参考图",
    category: "enhance",
    refRequired: false,
  },
  upscale: {
    id: "upscale",
    name: "高清放大 (Upscale)",
    short: "超清放大",
    icon: Sparkles,
    desc: "提升分辨率与细节",
    scenario: "最终出图",
    category: "enhance",
    refRequired: false,
  },
  img2video: {
    id: "img2video",
    name: "图生视频 (I2V)",
    short: "生视频",
    icon: Clapperboard,
    desc: "静态图片转动态短视频",
    scenario: "电商动态详情 / 社交媒体",
    refLabel: "尾帧参考图",
    category: "video",
    refRequired: false,
  },
};

const PROMPT_TEMPLATES = {
  bg_replace: {
    categories: [
      { name: "场景风格", key: "style", options: ["纯白摄影棚", "极简家居", "大理石台面", "清新自然户外", "高级展台", "赛博朋克"] },
      { name: "光影氛围", key: "vibe", options: ["柔和明亮", "自然光", "专业布光", "电影感", "暖色调", "冷淡风"] },
    ],
  },
  gesture_swap: {
    categories: [{ name: "手势类型", key: "style", options: ["单手握持", "指尖捏住", "双手捧起", "手掌展示", "使用中(涂抹)"] }],
  },
  product_swap: {
    categories: [{ name: "商品材质", key: "style", options: ["哑光质感", "亮面反光", "透明玻璃", "金属光泽", "磨砂表面"] }],
  },
  relight: {
    categories: [
      { name: "布光类型", key: "style", options: ["柔和漫射光(Soft)", "伦勃朗光(Rembrandt)", "强对比侧光(Hard Side)", "自然窗光(Window)", "蝴蝶光(Butterfly)", "赛博霓虹(Neon)"] },
      { name: "光源位置", key: "direction", options: ["左侧光", "右侧光", "顶光", "逆光(Backlight)", "正面平光"] },
      { name: "色温/氛围", key: "vibe", options: ["暖色调(Warm)", "冷色调(Cool)", "中性白(Neutral)", "夕阳感(Sunset)", "清晨感(Morning)"] },
    ],
  },
  img2video: {
    categories: [
      { name: "画幅比例", key: "ratio", options: ["16:9", "9:16", "3:4", "adaptive"] },
    ],
  },
};

const ASPECT_RATIOS = [
  { label: "1:1", w: 24, h: 24 },
  { label: "4:3", w: 32, h: 24 },
  { label: "3:4", w: 24, h: 32 },
  { label: "16:9", w: 40, h: 22 },
  { label: "9:16", w: 22, h: 40 },
];

const extractApiError = (data) => {
  const d = data?.detail ?? data?.message ?? data;
  if (typeof d === "string") return d;
  if (Array.isArray(d)) return d.map((x) => x?.msg || JSON.stringify(x)).join(" ; ");
  if (d && typeof d === "object") return JSON.stringify(d);
  return String(d);
};

const isVideoContent = (url) => {
  if (!url) return false;
  if (url.startsWith("data:video")) return true;
  if (url.includes(".mp4") || url.includes(".webm")) return true;
  return false;
};

// --- Helper: Graph Traversal ---
const getDownstreamNodes = (startNodeIds, nodes, connections) => {
  const visited = new Set(startNodeIds);
  const queue = [...startNodeIds];
  while (queue.length > 0) {
    const currentId = queue.shift();
    const outgoing = connections.filter((c) => c.from === currentId).map((c) => c.to);
    for (const nextId of outgoing) {
      if (!visited.has(nextId)) {
        visited.add(nextId);
        queue.push(nextId);
      }
    }
  }
  return visited;
};



// --- Helper: Node Ready Check ---
const checkNodeReady = (node, nodes, connections) => {
  if (node.type === NODE_TYPES.INPUT) return (node.data.images?.length || 0) > 0;
  if (node.type === NODE_TYPES.TEXT_INPUT) return (node.data.text?.length || 0) > 0;
  if (node.type === NODE_TYPES.OUTPUT) return true;

  const inputConns = connections.filter((c) => c.to === node.id);
  const sourceNodes = inputConns.map((c) => nodes.find((n) => n.id === c.from)).filter(Boolean);

  const hasUpstreamImages = sourceNodes.some((n) => (n.data.images?.length || 0) > 0 || (n.data.uploadedImages?.length || 0) > 0);
  const hasUpstreamText = sourceNodes.some((n) => (n.data.text?.length || 0) > 0);
  const hasLocalImages = (node.data.uploadedImages?.length || 0) > 0;
  const hasInternalPrompt = node.data.prompt && node.data.prompt.length > 0;

  if (node.data.mode === "text2img") return hasUpstreamText || hasInternalPrompt;
  if (node.data.mode === "multi_image_generate") return hasUpstreamImages || hasLocalImages;
  if (node.data.mode === "img2video") return hasUpstreamImages;
  return hasUpstreamImages;
};

const VideoPlayer = ({ src, className, controls = false, autoPlay = true, ...props }) => {
  const [error, setError] = useState(false);
  if (error)
    return (
      <div className={`flex flex-col items-center justify-center bg-slate-900 text-slate-500 ${className}`}>
        <FileWarning className="w-6 h-6 mb-1 text-red-400" />
        <span className="text-[10px]">视频加载失败</span>
      </div>
    );
  return (
    <video
      src={src}
      className={className}
      controls={controls}
      autoPlay={autoPlay}
      loop
      muted
      playsInline
      crossOrigin="anonymous"
      onError={() => setError(true)}
      {...props}
    />
  );
};

const AgentInputBar = ({ onSend, isLoading }) => {
  const [text, setText] = useState("");

  const handleSend = () => {
    if (!text.trim() || isLoading) return;
    onSend(text);
    setText("");
  };

  return (
    <div className="absolute bottom-20 left-1/2 -translate-x-1/2 z-50 w-full max-w-lg animate-in slide-in-from-bottom-5 fade-in duration-300">
      <div className="relative group">
        <div className="absolute -inset-0.5 bg-gradient-to-r from-pink-600 to-purple-600 rounded-xl blur opacity-30 group-hover:opacity-60 transition duration-1000 group-hover:duration-200"></div>
        <div className="relative flex items-center bg-slate-900 border border-slate-700 rounded-xl shadow-2xl overflow-hidden">
          <div className="pl-3 pr-2 text-purple-400 animate-pulse">
            <Bot className="w-5 h-5" />
          </div>
          <input
            type="text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
            placeholder="输入指令，例如：生成一张赛博朋克风格的猫"
            className="flex-1 bg-transparent border-none outline-none text-sm text-white placeholder-slate-500 h-12"
            disabled={isLoading}
          />
          <button
            onClick={handleSend}
            disabled={!text.trim() || isLoading}
            className={`mr-1 p-2 rounded-lg transition-colors ${
              text.trim() && !isLoading ? "text-purple-400 hover:bg-purple-500/10" : "text-slate-600"
            }`}
          >
            {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
          </button>
        </div>
        <div className="absolute -top-3 left-3 px-2 py-0.5 bg-gradient-to-r from-purple-600 to-pink-600 rounded-full text-[9px] font-bold text-white shadow-lg pointer-events-none">
          AI Agent
        </div>
      </div>
    </div>
  );
};

const ToolIconBtn = ({ icon: Icon, onClick, disabled, active, title }) => (
  <button
    onClick={onClick}
    disabled={disabled}
    title={title}
    className={`p-1.5 rounded transition-colors ${
      disabled
        ? "text-slate-600 cursor-not-allowed"
        : active
        ? "bg-red-500/20 text-red-400 hover:bg-red-500/30"
        : "text-slate-500 hover:bg-slate-800 hover:text-slate-200"
    }`}
  >
    <Icon className="w-4 h-4" />
  </button>
);

const SidebarBtn = ({ icon: Icon, label, desc, onClick, color, bg }) => (
  <button
    onClick={onClick}
    className="flex items-center gap-3 p-2 bg-slate-800 hover:bg-slate-700 rounded-lg border border-slate-700 text-left transition-all hover:translate-x-1 group w-full"
  >
    <div className={`w-8 h-8 rounded-md ${bg} flex items-center justify-center ${color} shrink-0`}>
      <Icon className="w-5 h-5" />
    </div>
    <div className="flex flex-col overflow-hidden">
      <div className="font-medium text-xs text-slate-300 group-hover:text-white truncate">{label}</div>
      <div className="text-[10px] text-slate-500 truncate">{desc}</div>
    </div>
    <Plus className="w-3 h-3 ml-auto text-slate-600 opacity-0 group-hover:opacity-100 transition-opacity" />
  </button>
);




const PropertyPanel = ({ node, updateData, onClose }) => {
  const [showAdvanced, setShowAdvanced] = useState(false);

  if (!node || [NODE_TYPES.INPUT, NODE_TYPES.OUTPUT, NODE_TYPES.TEXT_INPUT].includes(node.type)) return null;

  const isProcessor = node.type === NODE_TYPES.PROCESSOR;
  const isPostProcessor = node.type === NODE_TYPES.POST_PROCESSOR;
  const isVideoGen = node.type === NODE_TYPES.VIDEO_GEN;

  const currentMode = TOOL_CARDS[node.data.mode] || TOOL_CARDS.bg_replace;
  const activeTemplates = PROMPT_TEMPLATES[node.data.mode];

  const theme = (() => {
    if (isPostProcessor) return { text: "text-cyan-400", bg: "bg-cyan-600", border: "border-cyan-500" };
    if (isVideoGen) return { text: "text-rose-400", bg: "bg-rose-600", border: "border-rose-500" };
    return { text: "text-purple-400", bg: "bg-purple-600", border: "border-purple-500" };
  })();

  const availableTools = Object.keys(TOOL_CARDS).filter((key) => {
    const tool = TOOL_CARDS[key];
    if (isProcessor) return tool.category === "generate";
    if (isPostProcessor) return tool.category === "enhance";
    if (isVideoGen) return tool.category === "video";
    return false;
  });

  const handleRefUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onloadend = () => updateData(node.id, { refImage: reader.result });
    reader.readAsDataURL(file);
  };


  const updateTemplateData = (key, value) => {
    // multi_image_generate 的 prompt 更像“主 prompt”
    const newTemplates = { ...(node.data.templates || {}), [key]: value };
    // ✅ img2video：note 就是主提示词（直接覆盖 prompt）
    if (node.data.mode === "img2video" && key === "note") {
      updateData(node.id, { templates: newTemplates, prompt: value });
      return;
    }

    const parts = [];
    if (newTemplates.style) parts.push(newTemplates.style);
    if (newTemplates.vibe) parts.push(newTemplates.vibe);
    if (newTemplates.direction) parts.push(newTemplates.direction);
    if (newTemplates.note) parts.push(newTemplates.note);

    // text2img / multi_image_generate：prompt 不强制拼接
    const autoPrompt = parts.filter(Boolean).join(", ");
    updateData(node.id, { templates: newTemplates, prompt: node.data.mode === "relight" ? autoPrompt : (node.data.prompt || autoPrompt) });
  };

  const previewPrompt = [
    node.data.prompt,
    node.data.templates?.style,
    node.data.templates?.direction,
    node.data.templates?.vibe,
    node.data.templates?.note,
  ]
    .filter(Boolean)
    .join(", ");

  return (
  <div className="w-80 bg-slate-900 border-l border-slate-800 z-40 flex flex-col shadow-xl shrink-0 h-full min-h-0 overflow-hidden animate-in slide-in-from-right duration-200">
    {/* Header（固定） */}
    <div className="flex items-center justify-between border-b border-slate-800 p-4">
      <div className="flex items-center gap-2">
        <Sliders className="w-4 h-4 text-slate-400" />
        <span className="font-bold text-sm text-slate-200">配置面板</span>
      </div>
      <button onClick={onClose} className="text-slate-500 hover:text-white p-1 rounded hover:bg-slate-800">
        <X className="w-4 h-4" />
      </button>
    </div>

    {/* 内容区（可滚动） */}
    <div className="flex-1 min-h-0 overflow-y-auto p-4 space-y-4 custom-scrollbar">
      {/* 基础设置 */}
      <div className="space-y-3">
        <div className="space-y-2">
          <div className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">模式选择</div>

          <div className="grid grid-cols-2 gap-2">
            {availableTools.map((key) => {
              const tool = TOOL_CARDS[key];
              const isActive = node.data.mode === key;
              if (key === "text2img" || key === "multi_image_generate") return null;

              return (
                <button
                  key={key}
                  onClick={() => updateData(node.id, { mode: key, prompt: "", templates: {} })}
                  className={`relative flex flex-col p-2 rounded-lg border text-left transition-all ${
                    isActive
                      ? `bg-opacity-10 ${theme.bg} ${theme.border} shadow-sm`
                      : "bg-slate-950 border-slate-800 hover:border-slate-600"
                  }`}
                >
                  <div className="flex items-center gap-2 mb-1">
                    <tool.icon className={`w-4 h-4 ${isActive ? theme.text : "text-slate-500"}`} />
                    <span className={`text-xs font-bold ${isActive ? theme.text : "text-slate-300"}`}>{tool.short}</span>
                  </div>
                </button>
              );
            })}
          </div>

          {isProcessor && (
            <div className="flex gap-2 pt-1">
              {["text2img", "multi_image_generate"].map((key) => {
                const tool = TOOL_CARDS[key];
                const isActive = node.data.mode === key;
                return (
                  <button
                    key={key}
                    onClick={() => updateData(node.id, { mode: key, prompt: "", templates: { size: "1024x1024", aspect_ratio: "1:1" } })}
                    className={`flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded-lg border text-[10px] transition-colors ${
                      isActive
                        ? "bg-purple-600/10 border-purple-500 text-purple-400"
                        : "bg-slate-950 border-slate-800 text-slate-500 hover:border-slate-600"
                    }`}
                  >
                    <tool.icon className="w-3 h-3" />
                    {tool.name}
                  </button>
                );
              })}
            </div>
          )}
        </div>

{isVideoGen && (
  <div className="space-y-1">
    <div className="flex justify-between items-center">
      <div className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">尾帧参考图</div>
      <span className="text-[9px] text-slate-500 bg-slate-800 px-1.5 rounded">可选</span>
    </div>

    <div
      className={`relative w-full rounded border border-dashed bg-slate-950/50 flex items-center justify-center group transition-colors ${
        node.data.refImage ? `h-32 border-rose-500/50` : `h-24 border-slate-700 hover:border-rose-500`
      }`}
    >
      {node.data.refImage ? (
        <>
          <img
            src={node.data.refImage}
            className="w-full h-full object-cover rounded opacity-90 hover:opacity-100 transition-opacity"
            alt=""
          />
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              updateData(node.id, { refImage: null });
            }}
            className="absolute top-1 right-1 bg-black/60 rounded-full p-1.5 hover:bg-red-500 z-20 transition-colors"
            title="移除尾帧"
          >
            <X className="w-3 h-3 text-white" />
          </button>
        </>
      ) : (
        <div className="flex flex-col items-center text-slate-500 text-center px-4 cursor-pointer relative">
          <ImagePlus className="w-6 h-6 mb-2 text-slate-600" />
          <span className="text-[11px] font-medium text-slate-400">点击上传尾帧</span>
          <span className="text-[9px] text-slate-600 mt-1">用于引导结尾画面（可不传）</span>
          <input
            type="file"
            accept="image/*"
            className="absolute inset-0 opacity-0 cursor-pointer"
            onChange={handleRefUpload}
          />
        </div>
      )}
    </div>
  </div>
)}
        

        {/* Prompt Note */}
        <div className="space-y-1">
          <div className="text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-1">
            {node.data.mode === "multi_image_generate" || node.data.mode === "text2img" ? "提示词 (Prompt)" : "补充描述 / Note"}
          </div>
          <textarea
            className={`w-full bg-slate-950 border rounded p-2 text-xs text-slate-200 outline-none resize-none transition-colors border-slate-800 focus:${theme.border}`}
            rows={3}
            placeholder={node.data.mode === "relight" ? "例如: 增加暖色调氛围..." : "输入额外指令..."}
            value={
              node.data.mode === "multi_image_generate" || node.data.mode === "text2img"
                ? (node.data.prompt || "")
                : (node.data.templates?.note || node.data.prompt || "")
            }
            onChange={(e) => {
              if (node.data.mode === "multi_image_generate" || node.data.mode === "text2img") updateData(node.id, { prompt: e.target.value });
              else updateTemplateData("note", e.target.value);
            }}
          />
        </div>
      </div>

      {/* 高级设置 */}
      <button
        onClick={() => setShowAdvanced(!showAdvanced)}
        className="flex items-center justify-between text-xs text-slate-400 bg-slate-800/50 p-2 rounded hover:bg-slate-800 mt-2"
        type="button"
      >
        <span>高级设置 (模型/尺寸/风格)</span>
        {showAdvanced ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
      </button>

      {showAdvanced && (
        <div className="space-y-4 animate-in slide-in-from-top-2 duration-200">
          {(isProcessor || isPostProcessor) && (
            <div className="space-y-2">
              <div className="text-[10px] font-bold text-slate-500 uppercase tracking-wider flex items-center justify-between">
                <span className="flex items-center gap-1">
                  <Cpu className="w-3 h-3" /> AI 模型
                </span>
              </div>

              <div className="grid grid-cols-1 gap-2">
                {AI_MODELS.map((m) => (
                  <button
                    key={m.id}
                    onClick={() => updateData(node.id, { model: m.id })}
                    className={`flex items-center gap-2 p-2 rounded-lg border text-xs transition-all text-left ${
                      node.data.model === m.id
                        ? "bg-indigo-600/20 border-indigo-500 text-indigo-300"
                        : "bg-slate-950 border-slate-800 text-slate-400 hover:border-slate-600"
                    }`}
                    type="button"
                  >
                    <div className={`w-2 h-2 rounded-full shrink-0 ${node.data.model === m.id ? "bg-indigo-400" : "bg-slate-600"}`} />
                    <div className="flex flex-col overflow-hidden">
                      <span className="truncate font-medium">{m.name}</span>
                      <span className="text-[9px] opacity-60 truncate">{m.vendor}</span>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}

          {isVideoGen && (
            <div className="space-y-2">
              <div className="text-[10px] font-bold text-slate-500 uppercase tracking-wider flex items-center justify-between">
                <span className="flex items-center gap-1">
                  <Cpu className="w-3 h-3" /> 视频模型
                </span>
              </div>
              
              <div className="grid grid-cols-1 gap-2">
                {VIDEO_MODELS.map((m) => (
                  <button
                    key={m.id}
                    onClick={() => {
                      const nextModel = m.id;
                      const prevT = node.data.templates || {};

                      // ✅ 切到 1.5：时长 4~12；分辨率仅 480/720；默认开声音
                      if (nextModel === VIDEO_MODEL_1_5) {
                        const prevRes = (prevT.resolution || "").toLowerCase();
                        const nextResolution = ["480p", "720p"].includes(prevRes) ? prevRes : "720p";

                        const d = parseInt(String(prevT.duration ?? 5), 10);
                        const nextDuration = Math.min(12, Math.max(4, isNaN(d) ? 5 : d));

                        updateData(node.id, {
                          model: nextModel,
                          templates: {
                            ...prevT,
                            resolution: nextResolution,
                            duration: nextDuration,
                            generate_audio_new: prevT.generate_audio_new ?? true,
                          },
                        });
                        return;
                      }

                      // ✅ 切到 1.0：时长 3~12；分辨率允许 1080p
                      const d = parseInt(String(prevT.duration ?? 5), 10);
                      const nextDuration = Math.min(12, Math.max(3, isNaN(d) ? 5 : d));

                      const prevRes = (prevT.resolution || "").toLowerCase();
                      const nextResolution = prevRes || "1080p"; // 保留已有设置，没有就给 1080p

                      updateData(node.id, {
                        model: nextModel,
                        templates: {
                          ...prevT,
                          duration: nextDuration,
                          resolution: nextResolution,
                        },
                      });
                    }}
                    className={`flex items-center gap-2 p-2 rounded-lg border text-xs transition-all text-left ${
                      node.data.model === m.id
                        ? "bg-rose-600/20 border-rose-500 text-rose-200"
                        : "bg-slate-950 border-slate-800 text-slate-400 hover:border-slate-600"
                    }`}
                    type="button"
                  >
                    <div className={`w-2 h-2 rounded-full shrink-0 ${node.data.model === m.id ? "bg-rose-300" : "bg-slate-600"}`} />
                    <div className="flex flex-col overflow-hidden">
                      <span className="truncate font-medium">{m.name}</span>
                      <span className="text-[9px] opacity-60 truncate">{m.vendor}</span>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}
          {isVideoGen && (
  <div className="space-y-2">
    <div className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">视频时长 (秒)</div>

    {node.data.model === VIDEO_MODEL_1_5 ? (
      <input
        type="number"
        min={4}
        max={12}
        step={1}
        value={parseInt(String(node.data.templates?.duration ?? 5), 10)}
        onChange={(e) => {
          const v = parseInt(e.target.value, 10);
          const clamped = Math.min(12, Math.max(4, isNaN(v) ? 5 : v));
          updateData(node.id, { templates: { ...(node.data.templates || {}), duration: clamped } });
        }}
        className="w-full bg-slate-950 border border-slate-800 rounded p-2 text-xs text-slate-200 outline-none"
      />
    ) : (
      <div className="grid grid-cols-3 gap-2">
        {[3, 5, 10].map((sec) => {
          const cur = parseInt(String(node.data.templates?.duration ?? 5), 10);
          const isSel = cur === sec;
          return (
            <button
              key={sec}
              type="button"
              onClick={() => updateData(node.id, { templates: { ...(node.data.templates || {}), duration: sec } })}
              className={`px-2 py-1.5 rounded-md text-[10px] border transition-all ${
                isSel ? "bg-rose-600 border-rose-500 text-white" : "bg-slate-900 border-slate-800 text-slate-400 hover:border-slate-600"
              }`}
            >
              {sec}秒
            </button>
          );
        })}
      </div>
    )}
  </div>
)}
{isVideoGen && (
  <div className="space-y-2">
    <div className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">分辨率</div>

    <div className="grid grid-cols-3 gap-2">
      {(node.data.model === VIDEO_MODEL_1_5 ? ["480p", "720p"] : ["480p", "720p", "1080p"]).map((r) => {
        const isSel = (node.data.templates?.resolution || "1080p") === r;
        const label = r.toUpperCase(); // 480P/720P/1080P
        return (
          <button
            key={r}
            onClick={() => updateData(node.id, { templates: { ...(node.data.templates || {}), resolution: r } })}
            className={`px-2 py-1.5 rounded-md text-[10px] border transition-all ${
              isSel ? "bg-rose-600 border-rose-500 text-white" : "bg-slate-900 border-slate-800 text-slate-400 hover:border-slate-600"
            }`}
          >
            {label}
          </button>
        );
      })}
    </div>
  </div>
)}


          {/* Size & Ratio */}
          {isProcessor && (node.data.mode === "text2img" || node.data.mode === "multi_image_generate") && (
            <>
              <div>
                <div className="text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-2">尺寸 (Size)</div>
                <div className="grid grid-cols-3 gap-1.5">
                  {["1k", "2k", "4k"].map((opt) => {
                    let value = opt;
                    if (opt === "1k") value = "1024x1024";
                    const isSelected = node.data.templates?.size === value || (!node.data.templates?.size && opt === "1k");
                    return (
                      <button
                        key={opt}
                        onClick={() => updateData(node.id, { templates: { ...(node.data.templates || {}), size: value } })}
                        className={`px-2 py-1.5 rounded-md text-[10px] border transition-all ${
                          isSelected
                            ? "bg-purple-600 border-purple-500 text-white"
                            : "bg-slate-900 border-slate-800 text-slate-400 hover:border-slate-600"
                        }`}
                        type="button"
                      >
                        {opt}
                      </button>
                    );
                  })}
                </div>
              </div>

              <div>
                <div className="text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-2">比例 (Ratio)</div>
                <div className="grid grid-cols-5 gap-2">
                  {ASPECT_RATIOS.map((ar) => {
                    const isSelected = (node.data.templates?.aspect_ratio || "1:1") === ar.label;
                    return (
                      <button
                        key={ar.label}
                        onClick={() => updateData(node.id, { templates: { ...(node.data.templates || {}), aspect_ratio: ar.label } })}
                        className={`flex flex-col items-center gap-1 p-1 rounded-md border transition-all ${
                          isSelected
                            ? "bg-purple-600 border-purple-500 text-white"
                            : "bg-slate-900 border-slate-800 text-slate-400 hover:border-slate-600 hover:bg-slate-800"
                        }`}
                        title={ar.label}
                        type="button"
                      >
                        <div
                          className={`border ${isSelected ? "border-white bg-white/20" : "border-slate-500 bg-slate-800"}`}
                          style={{ width: ar.w, height: ar.h, borderRadius: 2 }}
                        />
                        <span className="text-[9px] scale-90">{ar.label}</span>
                      </button>
                    );
                  })}
                </div>
              </div>
            </>
          )}

          {/* Style Templates */}
          {activeTemplates?.categories?.map((cat, idx) => (
            <div key={idx} className="space-y-1.5">
              <div className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">{cat.name}</div>
              <div className="flex flex-wrap gap-1.5">
                {cat.options.map((opt) => {
                  const isSelected = node.data.templates?.[cat.key] === opt;
                  return (
                    <button
                      key={opt}
                      onClick={() =>
                        updateData(node.id, {
                          templates: { ...(node.data.templates || {}), [cat.key]: isSelected ? "" : opt },
                        })
                      }
                      className={`px-2 py-1 rounded-md text-[10px] border transition-all ${
                        isSelected ? `${theme.bg} ${theme.border} text-white` : "bg-slate-900 border-slate-800 text-slate-400 hover:border-slate-600"
                      }`}
                      type="button"
                    >
                      {opt}
                    </button>
                  );
                })}
              </div>
            </div>
          ))}

          {/* Batch Size */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">生成数量</span>
              <span className={`text-xs font-mono ${theme.text}`}>{node.data.batchSize || 1} 次</span>
            </div>
            <input
              type="range"
              min="1"
              max="4"
              step="1"
              value={node.data.batchSize || 1}
              onChange={(e) => updateData(node.id, { batchSize: parseInt(e.target.value) })}
              className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer"
            />
          </div>
        </div>
      )}
    </div>

    {/* Preview（固定在底部） */}
    <div className="p-4 border-t border-slate-800">
      <div className="text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-1">Prompt 预览</div>
      <div className="text-[10px] text-slate-400 font-mono bg-black/20 p-2 rounded border border-slate-800/50 break-words">
        {previewPrompt || "(暂无内容)"}
      </div>
    </div>
  </div>
);};

const NodeComponent = ({
  node,
  selected,
  zoom,
  onMouseDown,
  updateData,
  onDelete,
  onConnectStart,
  onConnectEnd,
  onPreview,
  onContinue,
  isReady,
  onRetry,
  onSelectArtifact,
  activeArtifact,
  onIterateImg2Img
}) => {
  const [showCopied, setShowCopied] = useState(false);

  const handleFileUpload = (e) => {
    const files = Array.from(e.target.files || []);
    if (!files.length) return;

    Promise.all(
      files.map(
        (file) =>
          new Promise((resolve) => {
            const reader = new FileReader();
            reader.onloadend = () => resolve(reader.result);
            reader.readAsDataURL(file);
          })
      )
    ).then((newImages) => {
      const currentImages = node.data.images || [];
      updateData(node.id, { images: [...currentImages, ...newImages] });
    });

    e.target.value = "";
  };

  const removeImage = (index) => {
    const newImages = (node.data.images || []).filter((_, i) => i !== index);
    updateData(node.id, { images: newImages });
  };

  const downloadAll = () => {
    (node.data.images || []).forEach((img, i) => {
      const link = document.createElement("a");
      link.href = img;
      link.download = `batch_result_${i}.${isVideoContent(img) ? "mp4" : "png"}`;
      link.click();
    });
  };

  const copyDebugInfo = () => {
    const info = JSON.stringify(
      {
        id: node.id,
        type: node.type,
        mode: node.data.mode,
        status: node.data.status,
        error: node.data.error,
        time: new Date().toISOString(),
      },
      null,
      2
    );
    navigator.clipboard.writeText(info);
    setShowCopied(true);
    setTimeout(() => setShowCopied(false), 2000);
  };

  const isProcessor = node.type === NODE_TYPES.PROCESSOR;
  const isPostProcessor = node.type === NODE_TYPES.POST_PROCESSOR;
  const isVideoGen = node.type === NODE_TYPES.VIDEO_GEN;
  const isAI = isProcessor || isPostProcessor || isVideoGen;
  const isInput = node.type === NODE_TYPES.INPUT;
  const isOutput = node.type === NODE_TYPES.OUTPUT;

  let statusColor = "border-slate-800";
  if (node.data.status === "error") statusColor = "border-red-500 shadow-[0_0_10px_rgba(239,68,68,0.3)]";
  else if (node.data.status === "success") statusColor = "border-green-500/50";
  else if (selected) statusColor = "border-white ring-2 ring-blue-500 ring-offset-2 ring-offset-slate-900";

  let title = "Node";
  if (isInput) title = `输入 (${node.data.images?.length || 0})`;
  if (isOutput) title = `输出 (${node.data.images?.length || 0})`;
  if (isProcessor) title = TOOL_CARDS[node.data.mode]?.name || "AI 处理器";
  if (isPostProcessor) title = TOOL_CARDS[node.data.mode]?.name || "后期增强";
  if (isVideoGen) title = "视频生成";
  if (node.type === NODE_TYPES.TEXT_INPUT) title = "Prompt";

  const getThemeColor = () => {
    if (isPostProcessor) return { text: "text-cyan-400", icon: Palette };
    if (isVideoGen) return { text: "text-rose-400", icon: Film };
    if (isInput) return { text: "text-blue-400", icon: Images };
    if (isOutput) return { text: "text-green-400", icon: Download };
    return { text: "text-purple-400", icon: Wand2 };
  };
  const theme = getThemeColor();
  const Icon = theme.icon;

  const safeProgressWidth = (() => {
    const total = node.data.total || 0;
    const prog = node.data.progress || 0;
    if (!total) return "0%";
    return `${Math.min(100, (prog / total) * 100)}%`;
  })();

  const renderArtifactThumb = (img, i, meta = {}) => {
    const isActive = activeArtifact?.url === img;

    return (
      <div
        key={i}
        className={`aspect-square relative group rounded overflow-hidden bg-slate-950 border cursor-pointer ${
          isActive ? "border-yellow-400 ring-2 ring-yellow-400/40" : "border-slate-800"
        }`}
        onPointerDown={(e) => e.stopPropagation()}
        onMouseDown={(e) => e.stopPropagation()}
        onClick={(e) => {
          e.stopPropagation();
          // ✅ 点缩略图：只预览
          onPreview?.(img);
        }}
        title="点击预览"
      >
        {isVideoContent(img) ? (
          <video src={img} className="w-full h-full object-cover" muted loop playsInline />
        ) : (
          <img src={img} className="w-full h-full object-cover" alt="" />
        )}

        <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity bg-gradient-to-t from-black/60 via-transparent to-transparent pointer-events-none" />

        {/* ✅ 真正可点的“选中产物”按钮：只选中，不预览 */}
        <button
          type="button"
          className="nodrag absolute bottom-1 left-1 text-[9px] px-2 py-1 rounded bg-black/60 text-white opacity-0 group-hover:opacity-100 hover:bg-yellow-500/30 hover:text-yellow-200 transition"
          onPointerDown={(e) => e.stopPropagation()}
          onMouseDown={(e) => e.stopPropagation()}
          onClick={(e) => {
            e.stopPropagation();

            const next = {
              url: img,
              kind: isVideoContent(img) ? "video" : "image",
              fromNodeId: node.id,
              createdAt: Date.now(),
              meta,
            };

            // ✅ 再点一次同一个：取消选中
            onSelectArtifact?.(activeArtifact?.url === img ? null : next);
          }}

          title="选中为 Agent 上下文"
        >
          选中产物
        </button>

        {isActive && (
          <div className="absolute top-1 left-1">
            <CheckCircle2 className="w-4 h-4 text-yellow-300 drop-shadow" />
          </div>
        )}
      </div>
    );
  };

  return (
    <div
      className={`absolute w-[280px] rounded-2xl border bg-slate-900/95 backdrop-blur-md shadow-2xl flex flex-col transition-colors transition-shadow duration-200 ${statusColor}`}
      style={{ left: node.x, top: node.y }}
      onMouseDown={onMouseDown}
    >
      {/* Header */}
      <div
        className={`flex justify-between items-center p-3 border-b border-slate-800 bg-slate-950/50 rounded-t-2xl handle cursor-grab active:cursor-grabbing ${
          selected ? "bg-blue-900/20" : ""
        }`}
      >
        <div className="flex items-center gap-2 overflow-hidden">
          <Icon className={`w-4 h-4 ${theme.text}`} />
          <span className="font-semibold text-sm text-slate-200 truncate select-none">{title}</span>
          {isReady && (
            <div className="w-2 h-2 rounded-full bg-green-500 shadow-[0_0_5px_rgba(34,197,94,0.8)]" title="Ready to Run" />
          )}
        </div>
        <div className="flex gap-1">
          {node.data.status === "error" && (
            <button
              onMouseDown={(e) => e.stopPropagation()}
              onClick={(e) => {
                e.stopPropagation();
                onRetry?.();
              }}
              className="text-red-400 hover:bg-red-900/30 p-1 rounded"
              title="重试"
              type="button"
            >
              <RotateCcw className="w-3.5 h-3.5" />
            </button>
          )}

          {isAI && (
            <div className={`p-1 rounded ${selected ? "bg-slate-800 text-white" : "text-slate-600"}`}>
              <Settings2 className="w-3.5 h-3.5" />
            </div>
          )}

          <button
            onMouseDown={(e) => e.stopPropagation()}
            onClick={(e) => {
              e.stopPropagation();
              onDelete?.();
            }}
            className="text-slate-500 hover:text-red-400 hover:bg-slate-800 rounded p-1 transition-colors"
            type="button"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      <div className="p-3 space-y-3">
        {/* Error */}
        {node.data.status === "error" && (
          <div className="bg-red-950/40 border border-red-900/50 rounded p-2 text-xs text-red-300 flex flex-col gap-2 animate-in fade-in zoom-in-95">
            <div className="flex items-start gap-2">
              <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-red-500" />
              <span className="break-all font-mono">{node.data.error || "Unknown Error"}</span>
            </div>
            <div className="flex justify-end gap-2 border-t border-red-900/30 pt-1 mt-1">
              <button
                onMouseDown={(e) => e.stopPropagation()}
                onClick={(e) => {
                  e.stopPropagation();
                  copyDebugInfo();
                }}
                className="flex items-center gap-1 text-[9px] opacity-70 hover:opacity-100"
                type="button"
              >
                <Clipboard className="w-3 h-3" /> {showCopied ? "已复制!" : "复制调试信息"}
              </button>
            </div>
          </div>
        )}

        {/* AI nodes */}
        {isAI && (
          <div className="space-y-2">
            {node.data.status === "loading" && (
              <div className="space-y-1">
                <div className="flex justify-between text-[10px] text-slate-400">
                  <span className="flex items-center gap-1">
                    <Loader2 className="w-3 h-3 animate-spin" /> 处理中...
                  </span>
                  <span>
                    {node.data.progress || 0}/{node.data.total || 0}
                  </span>
                </div>
                <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                  <div className="h-full bg-blue-500 transition-all duration-300" style={{ width: safeProgressWidth }} />
                </div>
              </div>
            )}

            {/* Results */}
            {node.data.status === "success" && node.data.images && node.data.images.length > 0 ? (
              <div className="grid grid-cols-2 gap-1.5">
                {node.data.images.map((img, i) =>
                  renderArtifactThumb(img, i, { mode: node.data.mode, prompt: node.data.prompt, model: node.data.model })
                )}
              </div>
            ) : (
              !["loading", "error"].includes(node.data.status) && (
                <div className="flex flex-col items-center justify-center py-6 text-slate-600 bg-slate-950/50 rounded border border-dashed border-slate-800">
                  {isReady ? <Play className="w-6 h-6 mb-2 text-green-500/50" /> : <Icon className="w-6 h-6 mb-2 opacity-20" />}
                  <span className="text-[10px]">{isReady ? "准备就绪" : "等待连接..."}</span>
                </div>
              )
            )}

            {node.data.status === "success" && (isProcessor || isPostProcessor) && (
              <button
                onMouseDown={(e) => e.stopPropagation()}
                onClick={(e) => {
                  e.stopPropagation();
                  onContinue?.(node.id);
                }}
                className="w-full py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded text-[10px] flex items-center justify-center gap-1 transition-colors border border-slate-700"
                type="button"
              >
                <Film className="w-3 h-3" /> 生成视频 <ArrowRight className="w-3 h-3" />
              </button>
            )}
            {/* ✅ 文生图后：一键续上图生图分支 */}
            {node.data.status === "success" && isProcessor && node.data.mode === "text2img" && (
              <button
                onMouseDown={(e) => e.stopPropagation()}
                onClick={(e) => {
                  e.stopPropagation();
                  onIterateImg2Img?.(node.id);
                }}
                className="w-full py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded text-[10px] flex items-center justify-center gap-1 transition-colors border border-slate-700"
                title="先点缩略图选中你要迭代的产物，再点这里"
              >
                <ImageIcon className="w-3 h-3" /> 继续图生图 <ArrowRight className="w-3 h-3" />
              </button>
            )}
          </div>
        )}

        {/* Input */}
        {isInput && (
          <div className="relative bg-slate-950 rounded border border-slate-800 flex flex-col overflow-hidden group hover:border-blue-500 transition-colors h-32">
            <div className="flex-1 overflow-y-auto p-2 custom-scrollbar">
              {node.data.images?.length > 0 ? (
                <div className="grid grid-cols-3 gap-1">
                  {node.data.images.map((img, i) => (
                    <div key={i} className="aspect-square relative group/img">
                      {/* ✅ 点图只预览；选中用按钮 */}
                      <img
                        src={img}
                        className={`w-full h-full object-cover rounded cursor-pointer border ${
                          activeArtifact?.url === img ? "border-yellow-400 ring-2 ring-yellow-400/40" : "border-transparent"
                        }`}
                        onMouseDown={(e) => e.stopPropagation()}
                        onClick={(e) => {
                          e.stopPropagation();
                          onPreview?.(img);
                        }}
                        title="点击预览"
                        alt=""
                      />

                      <button
                        type="button"
                        className="nodrag absolute bottom-1 left-1 text-[9px] px-1.5 py-0.5 rounded bg-black/60 text-white opacity-0 group-hover/img:opacity-100 hover:bg-yellow-500/30 hover:text-yellow-200 transition"
                        onMouseDown={(e) => e.stopPropagation()}
                        onClick={(e) => {
                          e.stopPropagation();
                          onSelectArtifact?.({ url: img, kind: "image", fromNodeId: node.id, createdAt: Date.now(), meta: { mode: "input" } });
                        }}
                        title="选中为 Agent 上下文"
                      >
                        选中
                      </button>

                      <button
                        onMouseDown={(e) => e.stopPropagation()}
                        onClick={(e) => {
                          e.stopPropagation();
                          removeImage(i);
                        }}
                        className="absolute -top-1 -right-1 bg-red-500 rounded-full p-0.5 opacity-0 group-hover/img:opacity-100"
                        type="button"
                        title="删除"
                      >
                        <X className="w-2 h-2 text-white" />
                      </button>
                    </div>
                  ))}

                  <div className="aspect-square bg-slate-800 rounded flex items-center justify-center cursor-pointer hover:bg-slate-700 relative">
                    <Plus className="w-4 h-4 text-slate-400" />
                    <input type="file" multiple accept="image/*" className="absolute inset-0 opacity-0 cursor-pointer" onChange={handleFileUpload} />
                  </div>
                </div>
              ) : (
                <div className="h-full flex flex-col items-center justify-center text-slate-500 relative">
                  <Images className="w-6 h-6 mb-1 opacity-50" />
                  <span className="text-[10px]">点击上传</span>
                  <input type="file" multiple accept="image/*" className="absolute inset-0 opacity-0 cursor-pointer" onChange={handleFileUpload} />
                </div>
              )}
            </div>
          </div>
        )}

        {/* Output */}
        {isOutput && (
          <div
            className="relative min-h-[100px] max-h-[200px] overflow-y-auto bg-slate-950 rounded border border-slate-800 p-1 custom-scrollbar nodrag"
            onMouseDown={(e) => e.stopPropagation()}
          >
            {node.data.images?.length > 0 ? (
              <div className="grid grid-cols-2 gap-1">
                {node.data.images.map((img, i) => (
                  <div key={i} className="aspect-square relative">
                    {renderArtifactThumb(img, i, {
                      mode: node.data.mode,
                      prompt: node.data.prompt,
                      model: node.data.model,
                    })}
                  </div>
                ))}
              </div>
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-xs text-slate-600 py-4">
                <CheckSquare className="w-6 h-6 opacity-20" />
                <span>结果展示区</span>
              </div>
            )}

            {node.data.images?.length > 0 && (
              <button
                onMouseDown={(e) => e.stopPropagation()}
                onClick={(e) => {
                  e.stopPropagation();
                  downloadAll();
                }}
                className="w-full mt-2 py-1 bg-green-900/30 text-green-400 border border-green-800 rounded text-[10px] flex justify-center items-center gap-1 hover:bg-green-900/50"
                type="button"
              >
                <Download className="w-3 h-3" /> 下载全部
              </button>
            )}
          </div>
        )}

        {/* Text input */}
        {node.type === NODE_TYPES.TEXT_INPUT && (
          <textarea
            className="w-full bg-slate-950 border border-slate-800 rounded p-2 text-xs text-slate-200 outline-none resize-none nodrag"
            rows={3}
            placeholder="输入提示词..."
            value={node.data.text || ""}
            onChange={(e) => updateData(node.id, { text: e.target.value })}
            onMouseDown={(e) => e.stopPropagation()}
          />
        )}
      </div>

      {/* Ports */}
      <div className="absolute top-[52px] w-full flex justify-between px-0 pointer-events-none">
        {node.type !== NODE_TYPES.INPUT && node.type !== NODE_TYPES.TEXT_INPUT && (
          <div
            onMouseUp={onConnectEnd}
            className="w-3 h-3 bg-slate-400 border-2 border-slate-800 rounded-full -ml-1.5 pointer-events-auto cursor-crosshair hover:bg-white hover:scale-125 z-20 shadow-lg"
          />
        )}
        {node.type !== NODE_TYPES.OUTPUT && (
          <div
            onMouseDown={onConnectStart}
            className="w-3 h-3 bg-slate-400 border-2 border-slate-800 rounded-full -mr-1.5 pointer-events-auto cursor-crosshair hover:bg-white hover:scale-125 ml-auto z-20 shadow-lg"
          />
        )}
      </div>
    </div>
  );
};

const getOrCreateThreadId = () => {
  const key = "bananaflow_thread_id";
  const saved = localStorage.getItem(key);
  if (saved) return saved;
  const tid = "canvas_" + Math.random().toString(36).slice(2, 14);
  localStorage.setItem(key, tid);
  return tid;
};

const newCanvasId = () => "canvas_" + Math.random().toString(36).slice(2, 12);

const Workbench = () => {
  const { user, logout, apiFetch } = useAuth();
  const [nodes, setNodes] = useState([]);
  const [connections, setConnections] = useState([]);
  const [history, setHistory] = useState([]);
  const [historyStep, setHistoryStep] = useState(-1);
  const [viewport, setViewport] = useState({ x: 0, y: 0, zoom: 1 });

  const [selectedNodeIds, setSelectedNodeIds] = useState(new Set());
  const [selectedConnectionIds, setSelectedConnectionIds] = useState(new Set());
  const [activeNodeId, setActiveNodeId] = useState(null);

  const [isSpacePressed, setIsSpacePressed] = useState(false);
  const [interactionMode, setInteractionMode] = useState("idle");
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [initialNodePos, setInitialNodePos] = useState({});
  const [selectionBox, setSelectionBox] = useState(null);
  const [connectingSource, setConnectingSource] = useState(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });

  const [isRunning, setIsRunning] = useState(false);
  const [isAgentThinking, setIsAgentThinking] = useState(false);
  const [runScope, setRunScope] = useState("selected_downstream");
  const [apiStatus, setApiStatus] = useState("checking");
  const [globalError, setGlobalError] = useState(null);
  const [loadingTip, setLoadingTip] = useState("");
  const [previewImage, setPreviewImage] = useState(null);
  const [isDemoMode, setIsDemoMode] = useState(false);
  const [showHistoryPanel, setShowHistoryPanel] = useState(false);
  const [activeHistoryTab, setActiveHistoryTab] = useState("recent");
  const [apiHistory, setApiHistory] = useState([]);
  const [apiStats, setApiStats] = useState(null);
  const [runToast, setRunToast] = useState(null);

  const canvasRef = useRef(null);
  const nodesRef = useRef(nodes);
  const connectionsRef = useRef(connections);

  const newCanvas = () => {
    const id = newCanvasId();
    setCanvasId(id);

    // 你想“新画布”是空白还是模板？这里示例空白
    pushHistory();
    setNodes([]);
    setConnections([]);
    setSelectedNodeIds(new Set());
    setSelectedConnectionIds(new Set());
    setActiveArtifact(null);
    setViewport({ x: 0, y: 0, zoom: 1 });
  };

  const [threadId] = useState(getOrCreateThreadId);
  const [canvasId, setCanvasId] = useState(() => {
    // 如果你后续支持“画布列表/切换”，这里可以从 URL 参数取
    const saved = localStorage.getItem(CANVAS_KEY);
    return saved || newCanvasId();
  });

  useEffect(() => { nodesRef.current = nodes; }, [nodes]);
  useEffect(() => { connectionsRef.current = connections; }, [connections]);
  useEffect(() => {
    localStorage.setItem(CANVAS_KEY, canvasId);
  }, [canvasId]);
  const [activeArtifact, setActiveArtifact] = useState(null);

  const applyPatch = useCallback((patchOps) => {
  if (!Array.isArray(patchOps) || patchOps.length === 0) return;

  // 从 ref 拿最新快照，避免闭包/批处理导致的“旧状态”
  let nextNodes = Array.isArray(nodesRef.current) ? [...nodesRef.current] : [];
  let nextConns = Array.isArray(connectionsRef.current) ? [...connectionsRef.current] : [];

  let finalSelectedNodeIds = null;
  let finalSelectedConnectionIds = null;
  let finalViewport = null;

  const hasNode = (id) => nextNodes.some(n => n?.id === id);
  const hasConn = (id) => nextConns.some(c => c?.id === id);

  const removeNodeAndEdges = (nodeId) => {
    nextNodes = nextNodes.filter(n => n.id !== nodeId);
    nextConns = nextConns.filter(c => c.from !== nodeId && c.to !== nodeId);
  };

  const removeConnById = (connId) => {
    nextConns = nextConns.filter(c => c.id !== connId);
  };

  patchOps.forEach((op) => {
    if (!op || !op.op) return;

    switch (op.op) {
      case "add_node": {
        const node = op.node;
        if (!node?.id) return;
        // 避免重复 add
        if (!hasNode(node.id)) nextNodes.push(node);
        break;
      }

      case "add_connection": {
        const c = op.connection;
        if (!c?.id || !c.from || !c.to) return;
        if (!hasConn(c.id)) nextConns.push(c);
        break;
      }

      case "update_node": {
        const id = op.id;
        if (!id) return;
        nextNodes = nextNodes.map(n => {
          if (n.id !== id) return n;

          // 允许更新 x/y（可选）
          const x = op.x ?? n.x;
          const y = op.y ?? n.y;

          // 合并 data（浅合并，够用；如果你 data 很深可换 deep merge）
          const mergedData = { ...(n.data || {}), ...(op.data || {}) };

          return { ...n, x, y, data: mergedData };
        });
        break;
      }

      // ✅兼容：remove_* / delete_* 都支持
      case "remove_node":
      case "delete_node": {
        if (!op.id) return;
        removeNodeAndEdges(op.id);
        break;
      }

      case "remove_connection":
      case "delete_connection": {
        if (!op.id) return;
        removeConnById(op.id);
        break;
      }

      // 可选：移动节点（如果后端以后加 move_node）
      case "move_node": {
        const { id, x, y } = op;
        if (!id) return;
        nextNodes = nextNodes.map(n => (n.id === id ? { ...n, x: x ?? n.x, y: y ?? n.y } : n));
        break;
      }

      // 可选：替换连接（等价于 remove + add）
      case "replace_connection": {
        const { id, connection } = op;
        if (id) removeConnById(id);
        if (connection?.id && connection.from && connection.to && !hasConn(connection.id)) {
          nextConns.push(connection);
        }
        break;
      }

      case "select_nodes": {
        finalSelectedNodeIds = new Set(op.ids || []);
        finalSelectedConnectionIds = new Set();
        break;
      }

      case "set_viewport": {
        finalViewport = op.viewport || null;
        break;
      }

      default:
        break;
    }
  });

  // 最后一次性提交（更稳）
  setNodes(nextNodes);
  setConnections(nextConns);

  if (finalSelectedNodeIds) {
    setSelectedNodeIds(finalSelectedNodeIds);
    setSelectedConnectionIds(finalSelectedConnectionIds || new Set());
  }
  if (finalViewport) setViewport(finalViewport);
}, [setNodes, setConnections, setSelectedNodeIds, setSelectedConnectionIds, setViewport]);

  // Initialize
  useEffect(() => {
    apiFetch("/docs")
      .then((r) => (r.ok ? setApiStatus("online") : setApiStatus("offline")))
      .catch(() => setApiStatus("offline"));

    if (history.length === 0) {
      setHistory([{ nodes: [], connections: [] }]);
      setHistoryStep(0);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiFetch]);

  // Sync active node
  useEffect(() => {
    if (selectedNodeIds.size === 1) setActiveNodeId(Array.from(selectedNodeIds)[0]);
    else setActiveNodeId(null);
  }, [selectedNodeIds]);

  const pushHistory = useCallback(() => {
    const s = { nodes: JSON.parse(JSON.stringify(nodes)), connections: JSON.parse(JSON.stringify(connections)) };
    setHistory((p) => {
      const n = p.slice(0, historyStep + 1);
      n.push(s);
      if (n.length > 50) n.shift();
      return n;
    });
    setHistoryStep((p) => Math.min(p + 1, 49));
  }, [nodes, connections, historyStep]);

  const undo = () => {
    if (historyStep > 0) {
      const p = history[historyStep - 1];
      setNodes(p.nodes);
      setConnections(p.connections);
      setHistoryStep(historyStep - 1);
    }
  };

  const redo = () => {
    if (historyStep < history.length - 1) {
      const n = history[historyStep + 1];
      setNodes(n.nodes);
      setConnections(n.connections);
      setHistoryStep(historyStep + 1);
    }
  };

  const deleteSelection = () => {
    if (selectedNodeIds.size === 0 && selectedConnectionIds.size === 0) return;
    pushHistory();
    setNodes((p) => p.filter((n) => !selectedNodeIds.has(n.id)));
    setConnections((p) =>
      p.filter((c) => !selectedConnectionIds.has(c.id) && !selectedNodeIds.has(c.from) && !selectedNodeIds.has(c.to))
    );
    setSelectedNodeIds(new Set());
    setSelectedConnectionIds(new Set());
  };

  useEffect(() => {
    const kd = (e) => {
      if (["INPUT", "TEXTAREA"].includes(e.target.tagName)) return;
      switch (e.key.toLowerCase()) {
        case " ":
          if (!e.repeat) setIsSpacePressed(true);
          break;
        case "z":
          if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            e.shiftKey ? redo() : undo();
          }
          break;
        case "y":
          if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            redo();
          }
          break;
        case "a":
          if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            setSelectedNodeIds(new Set(nodes.map((n) => n.id)));
          }
          break;
        case "delete":
        case "backspace":
          deleteSelection();
          break;
        case "=":
          if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            zoomCanvas(0.2);
          }
          break;
        case "-":
          if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            zoomCanvas(-0.2);
          }
          break;
        case "0":
          if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            setViewport({ x: 0, y: 0, zoom: 1 });
          }
          break;
      }
    };
    const ku = (e) => {
      if (e.key === " ") setIsSpacePressed(false);
    };
    window.addEventListener("keydown", kd);
    window.addEventListener("keyup", ku);
    return () => {
      window.removeEventListener("keydown", kd);
      window.removeEventListener("keyup", ku);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodes, history, historyStep, selectedNodeIds, selectedConnectionIds]);

  const screenToCanvas = (sx, sy) => {
    const r = canvasRef.current?.getBoundingClientRect();
    return {
      x: (sx - (r ? r.left : 0) - viewport.x) / viewport.zoom,
      y: (sy - (r ? r.top : 0) - viewport.y) / viewport.zoom,
    };
  };

  const zoomCanvas = (d, c = { x: window.innerWidth / 2, y: window.innerHeight / 2 }) => {
    const z = Math.min(Math.max(viewport.zoom + d, MIN_ZOOM), MAX_ZOOM);
    const w = screenToCanvas(c.x, c.y);
    const r = canvasRef.current?.getBoundingClientRect();
    setViewport({
      x: c.x - (r ? r.left : 0) - w.x * z,
      y: c.y - (r ? r.top : 0) - w.y * z,
      zoom: z,
    });
  };

  const handleWheel = (e) => {
    if (e.ctrlKey || e.metaKey) e.preventDefault();
    zoomCanvas(-e.deltaY * 0.001, { x: e.clientX, y: e.clientY });
  };

  const handleCanvasMouseDown = (e) => {
    if (e.button === 1 || (e.button === 0 && (isSpacePressed || e.altKey || e.metaKey))) {
      setInteractionMode("panning");
      setDragStart({ x: e.clientX - viewport.x, y: e.clientY - viewport.y });
      return;
    }
    if (e.button === 0 && !isSpacePressed && !e.altKey && !e.metaKey) {
      if (!e.shiftKey && !e.ctrlKey) {
        setSelectedNodeIds(new Set());
        setSelectedConnectionIds(new Set());
      }
      const s = screenToCanvas(e.clientX, e.clientY);
      setSelectionBox({ startX: s.x, startY: s.y, curX: s.x, curY: s.y });
      setInteractionMode("selecting");
    }
  };

const handleNodeMouseDown = (e, nid) => {
  e.stopPropagation();

  // ✅ 如果点在 nodrag 区域：只做“选中”，不要进入拖拽
  const isNoDragZone = !!e.target.closest(".nodrag");

  const s = new Set(selectedNodeIds);
  if (e.shiftKey || e.ctrlKey) s.has(nid) ? s.delete(nid) : s.add(nid);
  else if (!s.has(nid)) { s.clear(); s.add(nid); }

  setSelectedNodeIds(s);
  setSelectedConnectionIds(new Set());

  if (isNoDragZone) {
    setInteractionMode("idle");
    return;
  }

  setInteractionMode("dragging_node");
  setDragStart({ x: e.clientX, y: e.clientY });

  const p = {};
  nodes.forEach(n => {
    if (s.has(n.id) || n.id === nid) p[n.id] = { x: n.x, y: n.y };
  });
  setInitialNodePos(p);
};

  const handleMouseMove = useCallback(
    (e) => {
      setMousePos({ x: e.clientX, y: e.clientY });
      if (interactionMode === "panning") setViewport({ ...viewport, x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
      else if (interactionMode === "dragging_node") {
        const dx = (e.clientX - dragStart.x) / viewport.zoom;
        const dy = (e.clientY - dragStart.y) / viewport.zoom;
        setNodes((p) =>
          p.map((n) => (initialNodePos[n.id] ? { ...n, x: initialNodePos[n.id].x + dx, y: initialNodePos[n.id].y + dy } : n))
        );
      } else if (interactionMode === "selecting") {
        const c = screenToCanvas(e.clientX, e.clientY);
        setSelectionBox((p) => ({ ...p, curX: c.x, curY: c.y }));
      }
    },
    [interactionMode, dragStart, viewport, initialNodePos]
  );

  const handleMouseUp = () => {
    if (interactionMode === "dragging_node") pushHistory();
    if (interactionMode === "selecting" && selectionBox) {
      const x1 = Math.min(selectionBox.startX, selectionBox.curX);
      const x2 = Math.max(selectionBox.startX, selectionBox.curX);
      const y1 = Math.min(selectionBox.startY, selectionBox.curY);
      const y2 = Math.max(selectionBox.startY, selectionBox.curY);
      const s = new Set(selectedNodeIds);
      nodes.forEach((n) => {
        if (n.x < x2 && n.x + 280 > x1 && n.y < y2 && n.y + 200 > y1) s.add(n.id);
      });
      setSelectedNodeIds(s);
    }
    setInteractionMode("idle");
    setSelectionBox(null);
    setConnectingSource(null);
  };

  const getCursor = () => (interactionMode === "panning" || isSpacePressed ? "grab" : interactionMode === "dragging_node" ? "grabbing" : "default");

  const addNode = (t) => {
    pushHistory();
    const id = generateId();

    const r = canvasRef.current?.getBoundingClientRect();
    const cx = r ? r.width / 2 : window.innerWidth / 2;
    const cy = r ? r.height / 2 : window.innerHeight / 2;
    const c = screenToCanvas(cx, cy);

    const d = {
      [NODE_TYPES.INPUT]: { images: [] },
      [NODE_TYPES.TEXT_INPUT]: { text: "" },
      [NODE_TYPES.PROCESSOR]: {
        mode: "bg_replace",
        prompt: "",
        templates: { style: "", vibe: "", note: "", size: "1024x1024", aspect_ratio: "1:1" },
        batchSize: 1,
        uploadedImages: [],
        status: "idle",
        refImage: null,
        model: "gemini-3-pro-image-preview",
      },
      [NODE_TYPES.POST_PROCESSOR]: {
        mode: "relight",
        prompt: "",
        templates: { style: "", vibe: "", direction: "", note: "" },
        batchSize: 1,
        status: "idle",
        refImage: null,
        model: "gemini-3-pro-image-preview",
      },
      [NODE_TYPES.VIDEO_GEN]: {
        mode: "img2video",
        prompt: "",
        templates: { motion: "", camera: "", duration: 5, resolution: "1080p", ratio: "16:9", note: "" ,generate_audio_new: true,},
        batchSize: 1,
        status: "idle",
        refImage: null,
        model: VIDEO_MODEL_1_0, // ✅ 默认 1.0
      },
      [NODE_TYPES.OUTPUT]: { images: [] },
    };

    const newNode = { id, type: t, x: c.x - 140, y: c.y - 100, data: d[t] };

    let newConnection = null;
    if (selectedNodeIds.size === 1) {
      const sourceId = Array.from(selectedNodeIds)[0];
      const sourceNode = nodes.find((n) => n.id === sourceId);
      const canOutput = sourceNode?.type !== NODE_TYPES.OUTPUT;
      const canInput = t !== NODE_TYPES.INPUT && t !== NODE_TYPES.TEXT_INPUT;
      if (sourceNode && canOutput && canInput) {
        newConnection = { id: generateId(), from: sourceId, to: id };
        newNode.x = sourceNode.x + 350;
        newNode.y = sourceNode.y;
      }
    }

    setNodes((p) => [...p, newNode]);
    if (newConnection) setConnections((p) => [...p, newConnection]);
    setSelectedNodeIds(new Set([id]));
  };

  const createText2ImgTemplate = () => {
    pushHistory();
    const n1 = { id: generateId(), type: NODE_TYPES.TEXT_INPUT, x: 100, y: 200, data: { text: "赛博朋克风格的未来城市街道，霓虹灯光" } };
    const n2 = { id: generateId(), type: NODE_TYPES.PROCESSOR, x: 500, y: 200, data: { mode: "text2img", prompt: "", templates: { size: "1024x1024", aspect_ratio: "1:1" }, batchSize: 1, status: "idle", model: "gemini-3-pro-image-preview" } };
    const n3 = { id: generateId(), type: NODE_TYPES.OUTPUT, x: 900, y: 200, data: { images: [] } };
    setNodes([n1, n2, n3]);
    setConnections([
      { id: generateId(), from: n1.id, to: n2.id },
      { id: generateId(), from: n2.id, to: n3.id },
    ]);
    setViewport({ x: 0, y: 0, zoom: 1 });
  };

  const createImg2ImgTemplate = () => {
    pushHistory();
    const n0 = { id: generateId(), type: NODE_TYPES.TEXT_INPUT, x: 100, y: 100, data: { text: "保持原图构图，转为水彩风格" } };
    const n1 = { id: generateId(), type: NODE_TYPES.INPUT, x: 100, y: 350, data: { images: [] } };
    const n2 = { id: generateId(), type: NODE_TYPES.PROCESSOR, x: 500, y: 200, data: { mode: "multi_image_generate", prompt: "", templates: { size: "1024x1024", aspect_ratio: "1:1", note: "" }, batchSize: 1, uploadedImages: [], status: "idle", model: "gemini-3-pro-image-preview" } };
    const n3 = { id: generateId(), type: NODE_TYPES.OUTPUT, x: 900, y: 200, data: { images: [] } };
    setNodes([n0, n1, n2, n3]);
    setConnections([
      { id: generateId(), from: n0.id, to: n2.id },
      { id: generateId(), from: n1.id, to: n2.id },
      { id: generateId(), from: n2.id, to: n3.id },
    ]);
    setViewport({ x: 0, y: 0, zoom: 1 });
  };

  const createImg2VideoTemplate = () => {
    pushHistory();
    const n1 = { id: generateId(), type: NODE_TYPES.INPUT, x: 100, y: 200, data: { images: [] } };
    const n2 = { id: generateId(), type: NODE_TYPES.VIDEO_GEN, x: 500, y: 200, data: { mode: "img2video",model: VIDEO_MODEL_1_0, prompt: "", templates: { motion: "标准(Standard)", camera: "推近(Zoom In)", duration: 5, resolution: "1080p", ratio: "16:9", note: "" ,generate_audio_new: true,}, batchSize: 1, status: "idle", refImage: null } };
    const n3 = { id: generateId(), type: NODE_TYPES.OUTPUT, x: 900, y: 200, data: { images: [] } };
    setNodes([n1, n2, n3]);
    setConnections([
      { id: generateId(), from: n1.id, to: n2.id },
      { id: generateId(), from: n2.id, to: n3.id },
    ]);
    setViewport({ x: 0, y: 0, zoom: 1 });
  };

  const autoLayout = () => {
    pushHistory();
    const sorted = [...nodes].sort((a, b) => {
      const order = { [NODE_TYPES.INPUT]: 0, [NODE_TYPES.TEXT_INPUT]: 0, [NODE_TYPES.PROCESSOR]: 1, [NODE_TYPES.POST_PROCESSOR]: 2, [NODE_TYPES.VIDEO_GEN]: 3, [NODE_TYPES.OUTPUT]: 4 };
      return order[a.type] - order[b.type];
    });
    const newNodes = sorted.map((n, i) => ({ ...n, x: 100 + i * 350, y: 200 + (i % 2) * 50 }));
    setNodes(newNodes);
  };

  const createConnectedVideoNode = (sourceNodeId) => {
    pushHistory();
    const sourceNode = nodes.find((n) => n.id === sourceNodeId);
    if (!sourceNode) return;
    const newNodeId = generateId();
    const newNode = {
      id: newNodeId,
      type: NODE_TYPES.VIDEO_GEN,
      x: sourceNode.x + 350,
      y: sourceNode.y,
      data: {
        mode: "img2video",
        model: VIDEO_MODEL_1_0,
        prompt: sourceNode.data.prompt || "",
        templates: { motion: "标准(Standard)", camera: "固定镜头(Fixed)", duration: 5,  resolution: "1080p", ratio: "16:9", note: "",generate_audio_new: true, },
        batchSize: 1,
        status: "idle",
        refImage: null,
      },
    };
    setNodes((prev) => [...prev, newNode]);
    setConnections((prev) => [...prev, { id: generateId(), from: sourceNodeId, to: newNodeId }]);
    setSelectedNodeIds(new Set([newNodeId]));
  };

  // ✅ 继续图生图：在“创意生成(text2img)”后，自动接：输入 -> 图生图 -> 输出
const createConnectedImg2ImgBranch = useCallback(
  (sourceNodeId) => {
    pushHistory();

    const sourceNode = nodesRef.current.find((n) => n.id === sourceNodeId);
    if (!sourceNode) return;

    const imgs = sourceNode.data.images || [];

    // 优先用你刚刚“点缩略图选中的产物”
    const picked =
      activeArtifact?.fromNodeId === sourceNodeId ? activeArtifact.url : (imgs[0] || null);

    if (!picked) return;

    const inId = generateId();
    const procId = generateId();
    const outId = generateId();

    // 放到源节点右下方，避免重叠
    const baseX = sourceNode.x + 350;
    const baseY = sourceNode.y + 240;

    const inputNode = {
      id: inId,
      type: NODE_TYPES.INPUT,
      x: baseX,
      y: baseY,
      data: { images: [picked] },
    };

    const img2imgNode = {
      id: procId,
      type: NODE_TYPES.PROCESSOR,
      x: baseX + 350,
      y: baseY,
      data: {
        mode: "multi_image_generate",
        prompt: "",
        templates: { size: "1024x1024", aspect_ratio: "1:1" },
        batchSize: 1,
        uploadedImages: [],
        status: "idle",
        refImage: null,
        model: "gemini-3-pro-image-preview",
      },
    };

    const outputNode = {
      id: outId,
      type: NODE_TYPES.OUTPUT,
      x: baseX + 700,
      y: baseY,
      data: { images: [] },
    };

    setNodes((prev) => [...prev, inputNode, img2imgNode, outputNode]);

    // 连起来：创意生成 -> 输入(锁定图) -> 图生图 -> 输出
    setConnections((prev) => [
      ...prev,
      { id: generateId(), from: sourceNodeId, to: inId },
      { id: generateId(), from: inId, to: procId },
      { id: generateId(), from: procId, to: outId },
    ]);

    setSelectedNodeIds(new Set([procId]));
    setSelectedConnectionIds(new Set());
  },
  [pushHistory, activeArtifact]
);

  // ✅ Agent
  const handleAgentCommand = async (prompt) => {
    if (!prompt.trim()) return;

    setIsAgentThinking(true);
    setRunToast({ message: "Agent 正在构思工作流...", type: "info" });

    try {
      const resp = await apiFetch(`/api/agent/plan`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          canvas_id: canvasId,
          thread_id: threadId, // ✅ 新增：稳定 thread_id
          current_nodes: nodesRef.current,
          current_connections: connectionsRef.current,
          selected_artifact: activeArtifact || null,
        }),
      });

      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) throw new Error(extractApiError(data));

      // patch 增量
      if (data?.patch) {
        pushHistory();
        applyPatch(data.patch);
        setRunToast({ message: data.summary || `已增量更新：${data.patch.length} 步`, type: "info" });
        return;
      }

      // 全量重建
      const plan = data;
      pushHistory();

      const newNodes = (plan.nodes || []).map((planNode) => {
        const nodeData = {
          ...planNode.data,
          status: "idle",
          images: [],
          progress: 0,
          total: 0,
        };

        if (planNode.type === NODE_TYPES.VIDEO_GEN) {
          nodeData.mode = nodeData.mode || "img2video";
          nodeData.templates = nodeData.templates || { motion: "标准(Standard)", camera: "固定镜头(Fixed)", duration: "5秒", resolution: "1080p", ratio: "16:9", note: "" };
          nodeData.model = nodeData.model || "Doubao-Seedance-1.0-pro"; // ✅
        }

        return { ...planNode, data: nodeData };
      });

      const validatedConnections = (plan.connections || [])
        .filter((c) => newNodes.some((n) => n.id === c.from_id) && newNodes.some((n) => n.id === c.to_id))
        .map((c) => ({ id: `conn_${Math.random().toString(36).substr(2, 9)}`, from: c.from_id, to: c.to_id }));

      setNodes(newNodes);
      setConnections(validatedConnections);

      if (newNodes.length > 0) {
        const firstProc = newNodes.find((n) => n.type === "processor") || newNodes[0];
        setViewport({ x: window.innerWidth / 2 - firstProc.x * 0.8, y: window.innerHeight / 2 - firstProc.y * 0.8, zoom: 0.8 });
      }

      setRunToast({ message: `Agent 已成功搭建：${newNodes.length}个节点`, type: "info" });
    } catch (e) {
      console.error("Agent Error:", e);
      setRunToast({ message: `Agent 构思失败：${e.message || "请重试"}`, type: "error" });
    } finally {
      setIsAgentThinking(false);
    }
  };

  const deleteNode = (id) => {
    pushHistory();
    setNodes((p) => p.filter((n) => n.id !== id));
    setConnections((p) => p.filter((c) => c.from !== id && c.to !== id));
    setSelectedNodeIds((p) => {
      const s = new Set(p);
      s.delete(id);
      return s;
    });
  };

  const updateNodeData = (id, d) => setNodes((p) => p.map((n) => (n.id === id ? { ...n, data: { ...n.data, ...d } } : n)));

  const startConnection = (e, nid) => {
    e.stopPropagation();
    const r = e.currentTarget.getBoundingClientRect();
    setConnectingSource({ nodeId: nid, x: r.left + r.width / 2, y: r.top + r.height / 2 });
  };

  const completeConnection = (e, tid) => {
    e.stopPropagation();
    if (connectingSource && connectingSource.nodeId !== tid && !connections.find((c) => c.from === connectingSource.nodeId && c.to === tid)) {
      setConnections([...connections, { id: generateId(), from: connectingSource.nodeId, to: tid }]);
    }
    setConnectingSource(null);
  };

  const fetchHistoryAndStats = async () => {
    try {
      const histResp = await apiFetch(`/api/history`);
      if (histResp.ok) setApiHistory(await histResp.json());
      const statsResp = await apiFetch(`/api/stats`);
      if (statsResp.ok) setApiStats(await statsResp.json());
    } catch (e) {
      console.error("Failed to fetch history/stats", e);
    }
  };

  useEffect(() => {
    if (showHistoryPanel) fetchHistoryAndStats();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showHistoryPanel]);

  const applyHistoryConfig = (item) => {
    const targetCategory = TOOL_CARDS[item.mode]?.category;
    const targetType = targetCategory === "enhance" ? NODE_TYPES.POST_PROCESSOR : targetCategory === "video" ? NODE_TYPES.VIDEO_GEN : NODE_TYPES.PROCESSOR;
    const targetNodeId = Array.from(selectedNodeIds).find((id) => nodes.find((n) => n.id === id)?.type === targetType);
    if (!targetNodeId) {
      alert(`请先在画布上选中一个匹配的节点，再点击复用。`);
      return;
    }
    pushHistory();
    updateNodeData(targetNodeId, { mode: item.mode, prompt: item.prompt, templates: { note: item.prompt } });
    setShowHistoryPanel(false);
  };

  const handleRunClick = () => {
    let targetNodes = [];
    const allAiNodes = nodes.filter((n) => [NODE_TYPES.PROCESSOR, NODE_TYPES.POST_PROCESSOR, NODE_TYPES.VIDEO_GEN].includes(n.type));

    if (runScope === "all") targetNodes = allAiNodes;
    else if (runScope === "selected") targetNodes = allAiNodes.filter((n) => selectedNodeIds.has(n.id));
    else if (runScope === "selected_downstream") {
      const startIds = Array.from(selectedNodeIds).filter((id) => allAiNodes.find((n) => n.id === id));
      if (startIds.length === 0) {
        setGlobalError("请先选中起始节点");
        return;
      }
      const downstreamIds = getDownstreamNodes(startIds, nodes, connections);
      targetNodes = allAiNodes.filter((n) => downstreamIds.has(n.id));
    }

    if (targetNodes.length === 0) {
      setGlobalError("没有可运行的节点");
      return;
    }

    let taskCount = 0;
    targetNodes.forEach((n) => (taskCount += n.data.batchSize || 1));
    setRunToast({ message: `准备执行：${targetNodes.length} 个节点 / 共 ${taskCount} 次生成任务`, type: "info" });
    setTimeout(() => setRunToast(null), 3000);
    executeFlow(new Set(targetNodes.map((n) => n.id)));
  };

  const executeFlow = async (specificNodesSet) => {
    setGlobalError(null);

    const baseNodes = nodesRef.current;
    const baseConnections = connectionsRef.current;

    const clone = (obj) => (typeof structuredClone === "function" ? structuredClone(obj) : JSON.parse(JSON.stringify(obj)));
    const runtimeNodes = new Map(baseNodes.map((n) => [n.id, clone(n)]));

    const applyNodeUpdate = (id, patch) => {
      const cur = runtimeNodes.get(id);
      if (!cur) return;
      cur.data = { ...cur.data, ...patch };
      runtimeNodes.set(id, cur);
      setNodes((prev) => prev.map((n) => (n.id === id ? { ...n, data: { ...n.data, ...patch } } : n)));
    };

    const targetIds = Array.from(specificNodesSet);
    const targetNodes = targetIds.map((id) => runtimeNodes.get(id)).filter(Boolean);
    if (targetNodes.length === 0) return;

    const targetSet = new Set(targetIds);
    const indeg = new Map();
    const adj = new Map();
    targetIds.forEach((id) => {
      indeg.set(id, 0);
      adj.set(id, []);
    });

    baseConnections.forEach((c) => {
      if (targetSet.has(c.from) && targetSet.has(c.to)) {
        indeg.set(c.to, (indeg.get(c.to) || 0) + 1);
        adj.get(c.from).push(c.to);
      }
    });

    const q = [];
    targetIds.forEach((id) => {
      if ((indeg.get(id) || 0) === 0) q.push(id);
    });

    const orderedIds = [];
    while (q.length) {
      const id = q.shift();
      orderedIds.push(id);
      for (const nxt of adj.get(id) || []) {
        indeg.set(nxt, indeg.get(nxt) - 1);
        if (indeg.get(nxt) === 0) q.push(nxt);
      }
    }

    const finalOrder = orderedIds.length === targetIds.length ? orderedIds : targetIds;

    setIsRunning(true);
    setLoadingTip(LOADING_TIPS[Math.floor(Math.random() * LOADING_TIPS.length)]);
    const tipInterval = setInterval(() => setLoadingTip(LOADING_TIPS[Math.floor(Math.random() * LOADING_TIPS.length)]), 3000);

    setNodes((prev) =>
      prev.map((n) =>
        specificNodesSet.has(n.id) ? { ...n, data: { ...n.data, status: "loading", error: null, progress: 0, total: 0 } } : n
      )
    );

    try {
      for (const nodeId of finalOrder) {
        const procNode = runtimeNodes.get(nodeId);
        if (!procNode) continue;

        const incomingConns = baseConnections.filter((c) => c.to === procNode.id);
        const sourceNodes = incomingConns.map((c) => runtimeNodes.get(c.from)).filter(Boolean);

        const outputConn = baseConnections.find((c) => c.from === procNode.id);
        const targetOutput = outputConn ? runtimeNodes.get(outputConn.to) : null;

        const batchSize = procNode.data.batchSize || 1;

        let inputImages = [];
        let sourceText = "";

        sourceNodes.forEach((sn) => {
          const snImages = sn.data.images || [];
          const snUploads = sn.data.uploadedImages || [];
          inputImages.push(...snImages, ...snUploads);
          if (sn.data.text) sourceText += sn.data.text + " ";
        });
        sourceText = sourceText.trim();

        const needsSingle = procNode.data.mode === "multi_image_generate" || procNode.data.mode === "text2img";
        const effectiveInputCount = needsSingle ? 1 : inputImages.length;

        if (effectiveInputCount === 0 && !needsSingle) {
          applyNodeUpdate(procNode.id, { status: "error", error: "溯源失败：未检测到输入图片（请确认上游节点已先产出 images，并且连线正确）" });
          continue;
        }

        const totalTasks = effectiveInputCount * batchSize;
        applyNodeUpdate(procNode.id, { status: "loading", total: totalTasks, progress: 0 });

        const outputImages = [];

        for (let i = 0; i < effectiveInputCount; i++) {
          for (let b = 0; b < batchSize; b++) {
            try {
              let resultUrl = null;

              if (procNode.data.mode === "text2img") {
                const promptToUse = sourceText || procNode.data.prompt;
                if (!promptToUse?.trim()) throw new Error("缺少输入文本提示词");

                const resp = await apiFetch(`/api/text2img`, {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({
                    prompt: promptToUse,
                    model: procNode.data.model,
                    size: procNode.data.templates?.size || "1024x1024",
                    aspect_ratio: procNode.data.templates?.aspect_ratio || "1:1",
                  }),
                });
                const data = await resp.json();
                if (!resp.ok) throw new Error(extractApiError(data));
                if (data.images?.length > 0) resultUrl = data.images[0];
              } else if (procNode.type === NODE_TYPES.VIDEO_GEN || procNode.data.mode === "img2video") {
                const rawDuration = procNode.data.templates?.duration || "5";
                const durationInt = parseInt(String(rawDuration).replace(/[^0-9]/g, "")) || 5;
                const isCameraFixed = procNode.data.templates?.camera?.includes("固定") || false;

                const payload = {
                  model: procNode.data.model || "Doubao-Seedance-1.0-pro", // ✅ 新增
                  image: inputImages[i],
                  last_frame_image: procNode.data.refImage || null,
                  prompt: procNode.data.prompt || sourceText || "natural motion",
                  duration: durationInt,
                  fps: 24,
                  camera_fixed: isCameraFixed,
                  resolution: procNode.data.templates?.resolution || "1080p",
                  ratio: procNode.data.templates?.ratio || "16:9",
                  generate_audio: true, // ✅ 仅 1.5
                  seed: 21,
                };

                const resp = await apiFetch(`/api/img2video`, {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify(payload),
                });
                const data = await resp.json();
                if (!resp.ok) throw new Error(extractApiError(data));
                resultUrl = data.image; // 若后端返回字段不同（例如 data.video），在这里改
              } else if (procNode.data.mode === "multi_image_generate") {
                const resp = await apiFetch(`/api/multi_image_generate`, {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({
                    prompt: sourceText || procNode.data.prompt,
                    images: inputImages,
                    temperature: 0.7,
                    size: procNode.data.templates?.size || "1024x1024",
                    aspect_ratio: procNode.data.templates?.aspect_ratio || "1:1",
                  }),
                });
                const data = await resp.json();
                if (!resp.ok) throw new Error(extractApiError(data));
                resultUrl = data.image;
              } else {
                const payload = {
                  image: inputImages[i],
                  mode: procNode.data.mode,
                  prompt: procNode.data.prompt || sourceText,
                  ref_image: procNode.data.refImage,
                  model: procNode.data.model,
                };
                const resp = await apiFetch(`/api/edit`, {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify(payload),
                });
                const data = await resp.json();
                if (!resp.ok) throw new Error(extractApiError(data));
                resultUrl = data.image;
              }

              if (resultUrl) outputImages.push(resultUrl);
            } catch (e) {
              console.error("Task execution failed:", e);
              applyNodeUpdate(procNode.id, { status: "error", error: e.message || String(e) });
            }

            const currentProgress = i * batchSize + (b + 1);
            applyNodeUpdate(procNode.id, { progress: currentProgress });
          }
        }

        if (outputImages.length > 0) {
          applyNodeUpdate(procNode.id, { status: "success", images: outputImages });
          if (targetOutput && targetOutput.type === NODE_TYPES.OUTPUT) {
            const prevOut = targetOutput.data.images || [];
            applyNodeUpdate(targetOutput.id, { images: [...prevOut, ...outputImages] });
          }
        }
      }
    } finally {
      clearInterval(tipInterval);
      setIsRunning(false);
    }
  };

  const renderConnections = () =>
    connections.map((conn) => {
      const fromNode = nodes.find((n) => n.id === conn.from);
      const toNode = nodes.find((n) => n.id === conn.to);
      if (!fromNode || !toNode) return null;

      const x1 = fromNode.x + 280;
      const y1 = fromNode.y + 58;
      const x2 = toNode.x;
      const y2 = toNode.y + 58;

      const cp1x = x1 + (x2 - x1) / 2;
      const cp2x = x2 - (x2 - x1) / 2;
      const path = `M ${x1} ${y1} C ${cp1x} ${y1}, ${cp2x} ${y2}, ${x2} ${y2}`;

      return (
        <g key={conn.id}>
          <path d={path} stroke="transparent" strokeWidth="10" fill="none" className="cursor-pointer" />
          <path d={path} stroke={selectedConnectionIds.has(conn.id) ? "#fbbf24" : "#475569"} strokeWidth="2" fill="none" className="transition-colors duration-200" />
          {isRunning && (
            <circle r="3" fill="#fbbf24">
              <animateMotion dur="1.5s" repeatCount="indefinite" path={path} />
            </circle>
          )}
        </g>
      );
    });

  const renderTempConnection = () => {
    if (!connectingSource) return null;
    const n = nodes.find((nn) => nn.id === connectingSource.nodeId);
    if (!n) return null;

    const x1 = n.x + 280,
      y1 = n.y + 58;
    const target = screenToCanvas(mousePos.x, mousePos.y);
    const path = `M ${x1} ${y1} C ${x1 + (target.x - x1) / 2} ${y1}, ${target.x - (target.x - x1) / 2} ${target.y}, ${target.x} ${target.y}`;
    return <path d={path} stroke="#fbbf24" strokeWidth={2 / viewport.zoom} strokeDasharray="5,5" fill="none" />;
  };

  return (
    <div className="h-screen w-screen bg-slate-950 text-white overflow-hidden flex flex-col font-sans">
      <header className="h-14 bg-slate-900 border-b border-slate-800 flex items-center justify-between px-4 z-50 select-none shadow-md">
        <div className="flex items-center gap-3">
          <div className="bg-yellow-500/10 p-1.5 rounded-lg border border-yellow-500/20">
            <Zap className="text-yellow-400 w-5 h-5" />
          </div>
          <div className="flex flex-col">
            <span className="font-bold text-lg leading-tight tracking-tight">
              BananaFlow <span className="text-yellow-400">Workbench</span>
            </span>
            <span className="text-[10px] text-slate-400 font-medium">电商智能图像工作台</span>
          </div>
        </div>

        <div className="flex items-center gap-1 bg-slate-800 p-1 rounded-lg border border-slate-700">
          <ToolIconBtn icon={Undo} onClick={undo} disabled={historyStep <= 0} title="Undo (Ctrl+Z)" />
          <ToolIconBtn icon={Redo} onClick={redo} disabled={historyStep >= history.length - 1} title="Redo (Ctrl+Y)" />
          <div className="w-px h-4 bg-slate-700 mx-1"></div>
          <ToolIconBtn icon={Trash2} onClick={deleteSelection} title="Delete Selected" disabled={selectedNodeIds.size === 0 && selectedConnectionIds.size === 0} />
          <ToolIconBtn icon={Layout} onClick={autoLayout} title="Auto Layout" />
        </div>


        <div className="flex items-center gap-3">
          <div className={`flex items-center gap-1 px-2 py-1 rounded text-[10px] border transition-colors ${apiStatus === "online" ? "text-green-400 border-green-900/50 bg-green-900/20" : "text-red-400 border-red-900/50 bg-red-900/20"}`}>
            <Server className="w-3 h-3" /> {apiStatus === "online" ? "API Online" : "API Offline"}
          </div>

          <button
            onClick={() => setIsDemoMode(!isDemoMode)}
            className={`flex items-center gap-1 px-2 py-1 rounded text-[10px] border transition-colors ${isDemoMode ? "bg-blue-900/30 text-blue-300 border-blue-800" : "bg-slate-800 text-slate-500 border-transparent hover:border-slate-700"}`}
          >
            {isDemoMode ? <WifiOff className="w-3 h-3" /> : <Zap className="w-3 h-3 opacity-50" />} {isDemoMode ? "演示模式: ON" : "在线模式"}
          </button>

          <button
            onClick={newCanvas}
            className="px-2 py-1 rounded bg-slate-800 border border-slate-700 text-[10px] text-slate-300 hover:bg-slate-700"
          >
            新建画布
          </button>

          <div className="text-[10px] text-slate-500 font-mono">canvasId: {canvasId}</div>

          <div className="flex items-center gap-2 px-3 py-1 rounded-lg bg-slate-800 border border-slate-700">
            <div className="flex flex-col leading-tight">
              <span className="text-[11px] text-slate-200 font-semibold">{user?.email || "Guest"}</span>
              <span className="text-[10px] text-slate-500">已登录</span>
            </div>
            <button
              onClick={() => logout(true)}
              className="text-[10px] px-2 py-1 rounded bg-slate-900 border border-slate-700 text-slate-300 hover:text-white hover:border-slate-500"
            >
              Logout
            </button>
          </div>

          <div className="w-px h-6 bg-slate-800"></div>

          <div className="flex rounded-md shadow-lg shadow-purple-900/40">
            <button
              onClick={handleRunClick}
              disabled={isRunning}
              className={`flex items-center gap-2 px-4 py-1.5 rounded-l-md font-bold text-sm transition-all min-w-[100px] justify-center ${isRunning ? "bg-slate-700 cursor-not-allowed" : "bg-purple-600 hover:bg-purple-500"}`}
            >
              {isRunning ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />} <span className="truncate max-w-[150px]">{isRunning ? loadingTip : "运行"}</span>
            </button>
            <div className="relative group">
              <button className="px-2 py-1.5 bg-purple-700 rounded-r-md h-full hover:bg-purple-600 border-l border-purple-800">
                <ChevronDown className="w-4 h-4 text-purple-200" />
              </button>
              <div className="absolute right-0 top-full mt-1 w-32 bg-slate-800 border border-slate-700 rounded-lg shadow-xl overflow-hidden hidden group-hover:block z-50">
                <button onClick={() => setRunScope("all")} className={`w-full text-left px-3 py-2 text-xs hover:bg-slate-700 ${runScope === "all" ? "text-purple-400" : "text-slate-300"}`}>运行全部</button>
                <button onClick={() => setRunScope("selected")} className={`w-full text-left px-3 py-2 text-xs hover:bg-slate-700 ${runScope === "selected" ? "text-purple-400" : "text-slate-300"}`}>运行选中</button>
                <button onClick={() => setRunScope("selected_downstream")} className={`w-full text-left px-3 py-2 text-xs hover:bg-slate-700 ${runScope === "selected_downstream" ? "text-purple-400" : "text-slate-300"}`}>选中 → 下游</button>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Toast */}
      {runToast && (
        <div className={`fixed top-20 left-1/2 -translate-x-1/2 z-[60] px-4 py-2 rounded-lg shadow-2xl border flex items-center gap-2 animate-in slide-in-from-top-5 duration-300 ${runToast.type === "error" ? "bg-red-950/90 border-red-800 text-red-200" : "bg-slate-800/90 border-slate-600 text-white"}`}>
          {runToast.type === "error" ? <AlertCircle className="w-4 h-4" /> : <Activity className="w-4 h-4 text-purple-400" />}
          <span className="text-xs font-medium">{runToast.message}</span>
        </div>
      )}

      {/* Agent Input Bar */}
      <AgentInputBar onSend={handleAgentCommand} isLoading={isAgentThinking} />

      <div className="flex-1 flex relative">
        {/* Sidebar */}
        <div className="w-64 bg-slate-900 border-r border-slate-800 p-3 z-40 flex flex-col gap-2 shadow-xl select-none shrink-0">
          <div className="text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-1">工具箱</div>
          <SidebarBtn icon={MousePointer2} label="Prompt 输入" desc="纯文本提示词" onClick={() => addNode(NODE_TYPES.TEXT_INPUT)} color="text-yellow-400" bg="bg-yellow-500/10" />
          <SidebarBtn icon={Images} label="批量图片上传" desc="主商品图/素材" onClick={() => addNode(NODE_TYPES.INPUT)} color="text-blue-400" bg="bg-blue-500/10" />
          <SidebarBtn icon={Wand2} label="AI 处理器" desc="背景/手势/生成" onClick={() => addNode(NODE_TYPES.PROCESSOR)} color="text-purple-400" bg="bg-purple-500/10" />
          <SidebarBtn icon={Palette} label="后期增强" desc="光影精修/放大" onClick={() => addNode(NODE_TYPES.POST_PROCESSOR)} color="text-cyan-400" bg="bg-cyan-500/10" />
          <SidebarBtn icon={Film} label="视频生成" desc="图生视频/动效" onClick={() => addNode(NODE_TYPES.VIDEO_GEN)} color="text-rose-400" bg="bg-rose-500/10" />
          <SidebarBtn icon={Download} label="结果输出" desc="预览与下载" onClick={() => addNode(NODE_TYPES.OUTPUT)} color="text-green-400" bg="bg-green-500/10" />
        </div>

        {/* Canvas */}
        <div
          ref={canvasRef}
          className="flex-1 relative bg-slate-950 overflow-hidden"
          style={{ cursor: getCursor() }}
          onMouseDown={handleCanvasMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onWheel={handleWheel}
        >
          <div
            className="absolute inset-0 pointer-events-none opacity-20"
            style={{
              backgroundImage: `linear-gradient(#475569 1px, transparent 1px), linear-gradient(90deg, #475569 1px, transparent 1px)`,
              backgroundSize: `${GRID_SIZE * viewport.zoom}px ${GRID_SIZE * viewport.zoom}px`,
              backgroundPosition: `${viewport.x}px ${viewport.y}px`,
            }}
          />

          {nodes.length === 0 && (
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-10">
              <div className="bg-slate-900/80 border border-slate-700 p-8 rounded-2xl shadow-2xl backdrop-blur-sm text-center max-w-lg pointer-events-auto" onMouseDown={(e) => e.stopPropagation()}>
                <div className="flex justify-center mb-4">
                  <Zap className="w-12 h-12 text-yellow-400 bg-yellow-500/10 p-2 rounded-xl" />
                </div>
                <h2 className="text-xl font-bold text-white mb-2">欢迎来到 FlowStudio</h2>
                <p className="text-sm text-slate-400 mb-8">选择下方模版，或直接使用下方 AI 输入框：</p>

                <div className="grid grid-cols-3 gap-4">
                  <button onClick={createText2ImgTemplate} className="group flex flex-col items-center gap-3 p-4 bg-slate-800/50 hover:bg-slate-800 border border-slate-700 hover:border-purple-500/50 rounded-xl transition-all hover:-translate-y-1">
                    <div className="p-3 bg-purple-500/10 rounded-lg group-hover:bg-purple-500/20 transition-colors">
                      <Wand2 className="w-6 h-6 text-purple-400" />
                    </div>
                    <span className="text-xs font-bold text-slate-300 group-hover:text-white">文生图</span>
                  </button>

                  <button onClick={createImg2ImgTemplate} className="group flex flex-col items-center gap-3 p-4 bg-slate-800/50 hover:bg-slate-800 border border-slate-700 hover:border-blue-500/50 rounded-xl transition-all hover:-translate-y-1">
                    <div className="p-3 bg-blue-500/10 rounded-lg group-hover:bg-blue-500/20 transition-colors">
                      <Images className="w-6 h-6 text-blue-400" />
                    </div>
                    <span className="text-xs font-bold text-slate-300 group-hover:text-white">图生图</span>
                  </button>

                  <button onClick={createImg2VideoTemplate} className="group flex flex-col items-center gap-3 p-4 bg-slate-800/50 hover:bg-slate-800 border border-slate-700 hover:border-rose-500/50 rounded-xl transition-all hover:-translate-y-1">
                    <div className="p-3 bg-rose-500/10 rounded-lg group-hover:bg-rose-500/20 transition-colors">
                      <Clapperboard className="w-6 h-6 text-rose-400" />
                    </div>
                    <span className="text-xs font-bold text-slate-300 group-hover:text-white">图生视频</span>
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Controls */}
          <div className="absolute bottom-6 left-6 z-50 flex gap-2 select-none">
            <div className="bg-slate-800 border border-slate-700 rounded-lg flex items-center p-1 shadow-xl text-slate-500">
              <button onClick={() => zoomCanvas(-0.2)} className="p-2 hover:bg-slate-700 hover:text-slate-200 rounded transition-colors"><Minus className="w-4 h-4" /></button>
              <span className="w-12 text-center text-xs font-mono text-slate-400">{Math.round(viewport.zoom * 100)}%</span>
              <button onClick={() => zoomCanvas(0.2)} className="p-2 hover:bg-slate-700 hover:text-slate-200 rounded transition-colors"><Plus className="w-4 h-4" /></button>
            </div>
            <button onClick={() => setViewport({ x: 0, y: 0, zoom: 1 })} className="bg-slate-800 border border-slate-700 rounded-lg p-2 hover:bg-slate-700 hover:text-slate-200 shadow-xl text-slate-500" title="Reset View">
              <Maximize className="w-4 h-4" />
            </button>
            <div className="bg-slate-800 border border-slate-700 rounded-lg p-2 flex items-center gap-2 text-[10px] text-slate-500 shadow-xl">
              <Keyboard className="w-3 h-3" />
              <span>空格拖拽 | Delete 删除 | Ctrl+Z 撤销</span>
            </div>
          </div>

          <div className="absolute inset-0 origin-top-left" style={{ transform: `translate(${viewport.x}px,${viewport.y}px) scale(${viewport.zoom})` }}>
            <svg className="absolute inset-0 overflow-visible pointer-events-none" style={{ width: 1, height: 1 }}>
              {renderConnections()}
              {renderTempConnection()}
            </svg>

            {nodes.map((n) => (
              <NodeComponent
                key={n.id}
                node={n}
                selected={selectedNodeIds.has(n.id)}
                zoom={viewport.zoom}
                onMouseDown={(e) => handleNodeMouseDown(e, n.id)}
                updateData={updateNodeData}
                onDelete={() => deleteNode(n.id)}
                onConnectStart={(e) => {
                  e.stopPropagation();
                  pushHistory();
                  startConnection(e, n.id);
                }}
                onConnectEnd={(e) => {
                  e.stopPropagation();
                  pushHistory();
                  completeConnection(e, n.id);
                }}
                onPreview={setPreviewImage}
                onContinue={createConnectedVideoNode}
                onRetry={() => executeFlow(new Set([n.id]))}
                isReady={checkNodeReady(n, nodes, connections)}
                onSelectArtifact={setActiveArtifact} // ✅ 修复：现在生效
                activeArtifact={activeArtifact}
                onIterateImg2Img={createConnectedImg2ImgBranch}

              />
            ))}

            {selectionBox && (
              <div
                className="absolute border border-blue-500 bg-blue-500/20 pointer-events-none z-50"
                style={{
                  left: Math.min(selectionBox.startX, selectionBox.curX),
                  top: Math.min(selectionBox.startY, selectionBox.curY),
                  width: Math.abs(selectionBox.curX - selectionBox.startX),
                  height: Math.abs(selectionBox.curY - selectionBox.startY),
                }}
              />
            )}
          </div>
        </div>

        {/* ✅ 属性栏（保留并确保存在） */}
        <PropertyPanel node={activeNodeId ? nodes.find((n) => n.id === activeNodeId) : null} updateData={updateNodeData} onClose={() => setActiveNodeId(null)} />
      </div>

      {/* History Panel（保持你原逻辑） */}
      {showHistoryPanel && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/50 backdrop-blur-sm" onClick={() => setShowHistoryPanel(false)}>
          <div className="w-[600px] bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl flex flex-col max-h-[80vh] overflow-hidden" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between p-4 border-b border-slate-800">
              <div className="flex gap-4">
                <button onClick={() => setActiveHistoryTab("recent")} className={`text-sm font-bold pb-1 border-b-2 transition-colors ${activeHistoryTab === "recent" ? "border-purple-500 text-white" : "border-transparent text-slate-500"}`}>
                  最近任务
                </button>
                <button onClick={() => setActiveHistoryTab("stats")} className={`text-sm font-bold pb-1 border-b-2 transition-colors ${activeHistoryTab === "stats" ? "border-purple-500 text-white" : "border-transparent text-slate-500"}`}>
                  数据趋势
                </button>
              </div>
              <button onClick={() => setShowHistoryPanel(false)} className="text-slate-500 hover:text-white">
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="flex-1 overflow-y-auto p-4 custom-scrollbar bg-slate-950/50">
              {activeHistoryTab === "recent" ? (
                <div className="space-y-3">
                  {apiHistory.length === 0 && <div className="text-center text-slate-500 py-8">暂无历史记录</div>}
                  {apiHistory.map((item, i) => (
                    <div key={i} className="bg-slate-900 border border-slate-800 rounded-lg p-3 hover:border-purple-500/50 transition-colors flex justify-between items-start group">
                      <div className="flex-1 min-w-0 mr-4">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="text-xs font-bold text-purple-400 bg-purple-500/10 px-1.5 py-0.5 rounded uppercase">{TOOL_CARDS[item.mode]?.short || item.mode}</span>
                          <span className="text-[10px] text-slate-500">{new Date(item.time).toLocaleString()}</span>
                        </div>
                        <div className="text-xs text-slate-300 truncate font-mono">{item.prompt || "(无补充说明)"}</div>
                      </div>
                      <button onClick={() => applyHistoryConfig(item)} className="text-xs bg-slate-800 hover:bg-purple-600 text-slate-300 hover:text-white px-3 py-1.5 rounded transition-colors flex items-center gap-1 opacity-0 group-hover:opacity-100">
                        <RefreshCw className="w-3 h-3" /> 复用
                      </button>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="space-y-6">
                  <div>
                    <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">热门模式分布</h4>
                    <div className="space-y-2">
                      {apiStats &&
                        Object.entries(apiStats.modes || {}).map(([mode, count]) => (
                          <div key={mode} className="flex items-center gap-3">
                            <div className="w-24 text-xs text-slate-300 truncate text-right">{TOOL_CARDS[mode]?.short || mode}</div>
                            <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
                              <div className="h-full bg-blue-500" style={{ width: `${Math.min((count / 20) * 100, 100)}%` }} />
                            </div>
                            <div className="w-8 text-xs text-slate-500">{count}次</div>
                          </div>
                        ))}
                    </div>
                  </div>
                  <div>
                    <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">高频关键词 (灵感库)</h4>
                    <div className="flex flex-wrap gap-2">
                      {(apiStats?.keywords || []).map((word, i) => (
                        <span key={i} className="text-xs bg-slate-800 border border-slate-700 px-2 py-1 rounded-full text-slate-300 hover:text-white hover:border-blue-500 cursor-pointer transition-colors">
                          #{word}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Preview Modal */}
      {previewImage && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-sm p-10" onClick={() => setPreviewImage(null)}>
          <div className="relative max-w-full max-h-full flex items-center justify-center" onMouseDown={(e) => e.stopPropagation()}>
            {isVideoContent(previewImage) ? (
              <div className="relative" onClick={(e) => e.stopPropagation()}>
                <VideoPlayer src={previewImage} className="max-w-full max-h-[90vh] rounded-lg shadow-2xl border border-slate-700 bg-black" controls autoPlay />
                <button className="absolute -top-12 right-0 text-white/70 hover:text-white transition-colors bg-slate-800/50 p-2 rounded-full hover:bg-slate-700/80" onClick={() => setPreviewImage(null)}>
                  <X className="w-6 h-6" />
                </button>
              </div>
            ) : (
              <>
                <img src={previewImage} className="max-w-full max-h-[90vh] rounded-lg shadow-2xl border border-slate-700 object-contain" alt="Preview" onClick={(e) => e.stopPropagation()} />
                <button className="absolute -top-12 right-0 text-white/70 hover:text-white transition-colors bg-slate-800/50 p-2 rounded-full hover:bg-slate-700/80" onClick={() => setPreviewImage(null)}>
                  <X className="w-6 h-6" />
                </button>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default Workbench;
