import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { API_BASE, TOKEN_KEY } from "../config";

const AuthContext = createContext(null);
const API_ROOT = (API_BASE || "").replace(/\/+$/, "");

const buildUrl = (path) => {
  if (!path) return API_ROOT || "";
  if (path.startsWith("http")) return path;
  if (!API_ROOT) return path.startsWith("/") ? path : `/${path}`;
  return path.startsWith("/") ? `${API_ROOT}${path}` : `${API_ROOT}/${path}`;
};

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(() => localStorage.getItem(TOKEN_KEY));
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const clearSession = useCallback(() => {
    localStorage.removeItem(TOKEN_KEY);
    setToken(null);
    setUser(null);
  }, []);

  const apiFetch = useCallback(
    async (path, options = {}) => {
      const requestHeaders = new Headers(options.headers || {});
      if (token) requestHeaders.set("Authorization", `Bearer ${token}`);

      // ✅ 仅在“你确实传 JSON 字符串 body”时设置 Content-Type
      const body = options.body;
      const isFormData = typeof FormData !== "undefined" && body instanceof FormData;
      const isBlob = typeof Blob !== "undefined" && body instanceof Blob;

      if (!requestHeaders.has("Content-Type") && body && !isFormData && !isBlob) {
        if (typeof body === "string") requestHeaders.set("Content-Type", "application/json");
      }

      const resp = await fetch(buildUrl(path), { ...options, headers: requestHeaders });

      // ✅ 全局 401：清 session，避免“过期 token 还在跑请求”
      if (resp.status === 401) {
        clearSession();
      }

      return resp;
    },
    [token, clearSession],
  );

  const fetchProfile = useCallback(async () => {
    if (!token) {
      setLoading(false);
      return;
    }

    try {
      const resp = await apiFetch("/api/auth/me");
      if (!resp.ok) throw new Error("Failed to fetch profile");
      const data = await resp.json();
      setUser(data.user || null);
    } catch {
      clearSession();
    } finally {
      setLoading(false);
    }
  }, [apiFetch, clearSession, token]);

  useEffect(() => {
    fetchProfile();
  }, [fetchProfile]);

  const login = useCallback(async (email, password) => {
    setError(null);

    const resp = await fetch(buildUrl("/api/auth/login"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email: (email || "").trim(), password }),
    });

    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      const msg = data.detail || "登录失败";
      setError(msg);
      throw new Error(msg);
    }

    localStorage.setItem(TOKEN_KEY, data.access_token);
    setToken(data.access_token);
    setUser(data.user || null);
    return data;
  }, []);

  const register = useCallback(async (email, password) => {
    setError(null);

    const resp = await fetch(buildUrl("/api/auth/register"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email: (email || "").trim(), password }),
    });

    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      const msg = data.detail || "注册失败";
      setError(msg);
      throw new Error(msg);
    }

    localStorage.setItem(TOKEN_KEY, data.access_token);
    setToken(data.access_token);
    setUser(data.user || null);
    return data;
  }, []);

  const logout = useCallback(() => {
    clearSession();
  }, [clearSession]);

  const value = useMemo(
    () => ({ user, token, loading, error, login, register, logout, apiFetch }),
    [user, token, loading, error, login, register, logout, apiFetch],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
