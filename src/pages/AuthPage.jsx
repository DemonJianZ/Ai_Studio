import React, { useState } from "react";
import { useNavigate, Link } from "../router";
import { Zap, Shield, Mail, Lock, Loader2 } from "lucide-react";
import { useAuth } from "../auth/AuthProvider";

const gradientBg = "bg-[radial-gradient(circle_at_20%_20%,rgba(147,51,234,0.15),transparent_35%),radial-gradient(circle_at_80%_0%,rgba(56,189,248,0.1),transparent_25%),radial-gradient(circle_at_50%_80%,rgba(34,197,94,0.08),transparent_30%)]";

const inputStyle =
  "w-full bg-slate-900/60 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-purple-500 focus:ring-1 focus:ring-purple-500 transition";

const ENTERPRISE_DOMAIN = "dayukeji.com";
const PUBLIC_EMAIL_DOMAINS = new Set([
  "gmail.com",
  "outlook.com",
  "hotmail.com",
  "live.com",
  "yahoo.com",
  "icloud.com",
  "qq.com",
  "163.com",
  "126.com",
  "proton.me",
  "yeah.net",
]);

const getEmailDomain = (email) => {
  if (!email || !email.includes("@")) return "";
  return email.split("@").pop().toLowerCase();
};

export default function AuthPage({ mode = "login" }) {
  const navigate = useNavigate();
  const { login, register } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const emailDomain = getEmailDomain(email);
  const isPublicDomain = emailDomain && PUBLIC_EMAIL_DOMAINS.has(emailDomain);
  const isEnterpriseDomain = emailDomain === ENTERPRISE_DOMAIN;
  const isRegisterBlocked = !isEnterpriseDomain || isPublicDomain;

  const isLogin = mode === "login";

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      if (isLogin) {
        await login(email, password);
      } else {
        await register(email, password);
      }
      navigate("/app", { replace: true });
    } catch (err) {
      setError(err.message || "操作失败");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={`min-h-screen flex items-center justify-center bg-slate-950 text-white ${gradientBg}`}>
      <div className="absolute inset-0 overflow-hidden pointer-events-none opacity-60">
        <div className="absolute w-[600px] h-[600px] bg-purple-600/10 blur-3xl -top-40 -left-20" />
        <div className="absolute w-[500px] h-[500px] bg-cyan-500/10 blur-3xl top-10 right-0" />
      </div>

      <div className="relative w-full max-w-4xl grid grid-cols-1 md:grid-cols-2 gap-6 px-6">
        <div className="hidden md:flex flex-col justify-center rounded-2xl border border-slate-800/70 bg-slate-900/50 backdrop-blur-lg p-8 shadow-2xl shadow-purple-900/40">
          <div className="flex items-center gap-3 mb-6">
            <div className="bg-yellow-500/10 p-2 rounded-xl border border-yellow-500/20">
              <Zap className="text-yellow-400 w-6 h-6" />
            </div>
            <div>
              <div className="text-lg font-bold">BananaFlow Workbench</div>
              <div className="text-xs text-slate-400">AI 赋能的电商智能图像工作台</div>
            </div>
          </div>
          <div className="space-y-4 text-sm text-slate-300">
            <p className="flex items-center gap-2">
              <Shield className="w-4 h-4 text-purple-300" />
              注册/登录后可继续使用生成与编辑能力
            </p>
            <p className="flex items-center gap-2">
              <Shield className="w-4 h-4 text-purple-300" />
              登录态自动恢复，刷新无忧
            </p>
            <p className="flex items-center gap-2">
              <Shield className="w-4 h-4 text-purple-300" />
              当前阶段生成接口不扣费，放心体验
            </p>
          </div>
        </div>

      <div className="relative rounded-2xl border border-slate-800/70 bg-slate-900/60 backdrop-blur-lg p-8 shadow-2xl shadow-purple-900/40">
        <div className="flex items-center justify-between mb-8">
          <div>
            <div className="text-2xl font-bold mb-1">{isLogin ? "欢迎回来" : "创建账户"}</div>
            <div className="text-sm text-slate-400">{isLogin ? "登录后进入工作台" : "注册后自动登录"}</div>
            </div>
            <div className="px-3 py-1 rounded-full bg-purple-600/20 text-purple-200 text-xs border border-purple-500/20">
              AUTH · SECURE
            </div>
          </div>

          <form className="space-y-4" onSubmit={handleSubmit}>
            <label className="space-y-2 block">
              <span className="text-xs text-slate-400">Email</span>
              <div className="relative">
                <Mail className="w-4 h-4 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2" />
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  placeholder="you@example.com"
                  className={`${inputStyle} pl-10`}
                />
              </div>
            </label>

            {!isLogin && (
              <div className="text-xs">
                <div className={`mt-1 ${isPublicDomain || (emailDomain && !isEnterpriseDomain) ? "text-amber-300" : "text-slate-500"}`}>
                  仅支持使用公司邮箱（{ENTERPRISE_DOMAIN}）注册，公共邮箱将被拒绝。
                </div>
                {isPublicDomain && (
                  <div className="text-red-300 mt-1">检测到公共邮箱域名，请改用企业邮箱。</div>
                )}
              </div>
            )}

            <label className="space-y-2 block">
              <span className="text-xs text-slate-400">密码</span>
              <div className="relative">
                <Lock className="w-4 h-4 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2" />
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  minLength={6}
                  required
                  placeholder="至少6位"
                  className={`${inputStyle} pl-10`}
                />
              </div>
            </label>

            {error && <div className="text-red-400 text-xs bg-red-900/30 border border-red-800 rounded-lg px-3 py-2">{error}</div>}

            <button
              type="submit"
              disabled={loading || (!isLogin && isRegisterBlocked)}
              className="w-full py-2.5 rounded-lg bg-purple-600 hover:bg-purple-500 transition font-semibold text-sm flex items-center justify-center gap-2 shadow-lg shadow-purple-900/30 disabled:opacity-60"
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" /> 处理中...
                </>
              ) : (
                <>{isLogin ? "登录" : "注册并登录"}</>
              )}
            </button>
          </form>

          <div className="text-xs text-slate-400 mt-6 text-center">
            {isLogin ? "还没有账号？" : "已经有账号？"}
            <Link to={isLogin ? "/register" : "/login"} className="text-purple-300 font-semibold ml-2 hover:text-purple-200">
              {isLogin ? "立即注册" : "去登录"}
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
