import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { API_BASE, TOKEN_KEY } from "../config";

const AuthContext = createContext(null);

const buildUrl = (path) => {
  if (!path) return API_BASE;
  if (path.startsWith("http")) return path;
  if (path.startsWith("/")) return `${API_BASE}${path}`;
  return `${API_BASE}/${path}`;
};

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(() => localStorage.getItem(TOKEN_KEY));
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const clearSession = useCallback((redirect = true) => {
    localStorage.removeItem(TOKEN_KEY);
    setToken(null);
    setUser(null);
    if (redirect) {
      window.location.href = "/login";
    }
  }, []);

  const handleUnauthorized = useCallback(() => {
    clearSession(true);
  }, [clearSession]);

  const apiFetch = useCallback(
    async (path, options = {}) => {
      const requestHeaders = new Headers(options.headers || {});
      if (token) {
        requestHeaders.set("Authorization", `Bearer ${token}`);
      }
      const response = await fetch(buildUrl(path), {
        ...options,
        headers: requestHeaders,
      });
      if (response.status === 401) {
        handleUnauthorized();
      }
      return response;
    },
    [handleUnauthorized, token],
  );

  const fetchProfile = useCallback(async () => {
    if (!token) {
      setLoading(false);
      return;
    }
    try {
      const resp = await apiFetch("/api/auth/me");
      if (!resp.ok) {
        throw new Error("Unauthorized");
      }
      const data = await resp.json();
      setUser(data.user);
    } catch (err) {
      clearSession(false);
    } finally {
      setLoading(false);
    }
  }, [apiFetch, clearSession, token]);

  useEffect(() => {
    fetchProfile();
  }, [fetchProfile]);

  const login = useCallback(
    async (email, password) => {
      setError(null);
      const resp = await fetch(buildUrl("/api/auth/login"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) {
        throw new Error(data.detail || "登录失败");
      }
      localStorage.setItem(TOKEN_KEY, data.access_token);
      setToken(data.access_token);
      setUser(data.user);
      return data;
    },
    [],
  );

  const register = useCallback(
    async (email, password) => {
      setError(null);
      const resp = await fetch(buildUrl("/api/auth/register"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) {
        throw new Error(data.detail || "注册失败");
      }
      localStorage.setItem(TOKEN_KEY, data.access_token);
      setToken(data.access_token);
      setUser(data.user);
      return data;
    },
    [],
  );

  const logout = useCallback(
    (redirect = true) => {
      clearSession(redirect);
    },
    [clearSession],
  );

  const value = useMemo(
    () => ({ user, token, loading, error, login, register, logout, apiFetch }),
    [apiFetch, error, loading, login, logout, register, token, user],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
