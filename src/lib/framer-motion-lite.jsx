import React, { useEffect, useMemo, useRef } from "react";

const cubicBezier = (ease) => {
  if (Array.isArray(ease) && ease.length === 4) {
    return `cubic-bezier(${ease.join(",")})`;
  }
  if (typeof ease === "string") return ease;
  return undefined;
};

const buildKeyframes = (animate) => {
  if (!animate) return null;
  const keys = Object.keys(animate);
  if (keys.length === 0) return null;
  const frameCount = Array.isArray(animate[keys[0]]) ? animate[keys[0]].length : 1;
  return Array.from({ length: frameCount }, (_, idx) => {
    const frame = {};
    const transforms = [];
    keys.forEach((key) => {
      const value = animate[key];
      const current = Array.isArray(value) ? value[idx] : value;
      if (current === undefined) return;
      if (key === "x") transforms.push(`translateX(${current}px)`);
      else if (key === "y") transforms.push(`translateY(${current}px)`);
      else if (key === "scale") transforms.push(`scale(${current})`);
      else if (key === "scaleX") transforms.push(`scaleX(${current})`);
      else if (key === "scaleY") transforms.push(`scaleY(${current})`);
      else if (key === "rotate") transforms.push(`rotate(${current}deg)`);
      else frame[key] = current;
    });
    if (transforms.length) frame.transform = transforms.join(" ");
    return frame;
  });
};

const initialStyle = (initial) => {
  if (!initial) return {};
  const style = { ...initial };
  const transforms = [];
  if (initial.x !== undefined) transforms.push(`translateX(${initial.x}px)`);
  if (initial.y !== undefined) transforms.push(`translateY(${initial.y}px)`);
  if (initial.scale !== undefined) transforms.push(`scale(${initial.scale})`);
  if (initial.scaleX !== undefined) transforms.push(`scaleX(${initial.scaleX})`);
  if (initial.scaleY !== undefined) transforms.push(`scaleY(${initial.scaleY})`);
  if (initial.rotate !== undefined) transforms.push(`rotate(${initial.rotate}deg)`);
  if (transforms.length) {
    delete style.x;
    delete style.y;
    delete style.scale;
    delete style.scaleX;
    delete style.scaleY;
    delete style.rotate;
    style.transform = transforms.join(" ");
  }
  return style;
};

const createMotionComponent = (Component) => {
  const MotionComponent = React.forwardRef(
    ({ initial, animate, transition = {}, style, children, ...rest }, forwardedRef) => {
      const localRef = useRef(null);
      const combinedRef = forwardedRef || localRef;
      const keyframes = useMemo(() => buildKeyframes(animate), [animate]);
      const baseStyle = useMemo(() => ({ ...initialStyle(initial), ...style }), [initial, style]);

      useEffect(() => {
        const element = combinedRef && "current" in combinedRef ? combinedRef.current : null;
        if (!element || !keyframes) return undefined;
        const {
          duration = 1,
          delay = 0,
          repeat = 0,
          repeatType,
          repeatDelay = 0,
          ease,
        } = transition;

        const animation = element.animate(keyframes, {
          duration: duration * 1000,
          delay: delay * 1000,
          easing: cubicBezier(ease),
          iterations: repeat === Infinity ? Infinity : repeat ? repeat + 1 : 1,
          direction: repeatType === "mirror" ? "alternate" : repeatType === "reverse" ? "alternate-reverse" : "normal",
          endDelay: repeatDelay * 1000,
          fill: "forwards",
        });

        return () => animation.cancel();
      }, [combinedRef, keyframes, transition]);

      return (
        <Component ref={combinedRef} style={baseStyle} {...rest}>
          {children}
        </Component>
      );
    }
  );

  MotionComponent.displayName = `motion-${Component}`;
  return MotionComponent;
};

export const motion = {
  div: createMotionComponent("div"),
  svg: createMotionComponent("svg"),
  path: createMotionComponent("path"),
  circle: createMotionComponent("circle"),
};

export default motion;
