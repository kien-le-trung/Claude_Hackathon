"use client";

import { Grid, Line, OrbitControls, PerspectiveCamera } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { useEffect, useMemo, useState } from "react";

type Point = [number, number, number];
type SkeletonFrame = { timestamp_ms: number; points: Point[] | null };
export type SkeletonData = {
  schema_version: number;
  fps: number;
  smoothing: {
    method: string;
    window_length: number;
    polynomial_order: number;
  };
  connections: [number, number][];
  frames: SkeletonFrame[];
};

function displayPoint(point: Point): Point {
  return [point[0], point[2], -point[1]];
}

export function pointsAtTime(data: SkeletonData, seconds: number): Point[] | null {
  const frames = data.frames;
  if (!frames.length) return null;
  const timestamp = Math.max(0, seconds * 1000);
  let low = 0;
  let high = frames.length - 1;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (frames[middle].timestamp_ms < timestamp) low = middle + 1;
    else high = middle;
  }
  const nextIndex = low;
  if (frames[nextIndex].timestamp_ms <= timestamp || nextIndex === 0) {
    return frames[nextIndex].points;
  }
  const previous = frames[nextIndex - 1];
  const next = frames[nextIndex];
  if (!previous.points || !next.points) return null;
  const span = next.timestamp_ms - previous.timestamp_ms;
  if (span <= 0) return next.points;
  const amount = (timestamp - previous.timestamp_ms) / span;
  return previous.points.map((point, index) => [
    point[0] + (next.points![index][0] - point[0]) * amount,
    point[1] + (next.points![index][1] - point[1]) * amount,
    point[2] + (next.points![index][2] - point[2]) * amount,
  ]);
}

function Skeleton({
  points,
  connections,
}: {
  points: Point[];
  connections: [number, number][];
}) {
  const displayed = useMemo(() => points.map(displayPoint), [points]);
  return (
    <group>
      {connections.map(([start, end]) => (
        <Line
          key={`${start}-${end}`}
          points={[displayed[start], displayed[end]]}
          color="#32175b"
          lineWidth={4}
        />
      ))}
      {displayed.map((position, index) => (
        <mesh key={index} position={position}>
          <sphereGeometry args={[0.025, 16, 12]} />
          <meshStandardMaterial color="#7d3ceb" roughness={0.55} />
        </mesh>
      ))}
    </group>
  );
}

export default function SkeletonViewer({
  url,
  currentTime,
  onLoaded,
}: {
  url: string;
  currentTime: number;
  onLoaded: (duration: number) => void;
}) {
  const [data, setData] = useState<SkeletonData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [viewKey, setViewKey] = useState(0);

  useEffect(() => {
    let active = true;
    fetch(url, { cache: "no-store" })
      .then(async (response) => {
        if (!response.ok) throw new Error("Interactive skeleton unavailable");
        return response.json() as Promise<SkeletonData>;
      })
      .then((payload) => {
        if (!active) return;
        setData(payload);
        const last = payload.frames.at(-1);
        onLoaded(last ? last.timestamp_ms / 1000 : 0);
      })
      .catch((reason: unknown) => {
        if (active) setError(reason instanceof Error ? reason.message : "Could not load skeleton");
      });
    return () => { active = false; };
  }, [url, onLoaded]);

  const points = data ? pointsAtTime(data, currentTime) : null;

  return (
    <div className="skeleton-viewer">
      {error ? (
        <div className="viewer-status">{error}</div>
      ) : !data ? (
        <div className="viewer-status">Loading interactive skeleton…</div>
      ) : (
        <>
          <Canvas shadows dpr={[1, 2]}>
            <color attach="background" args={["#f8f5fc"]} />
            <PerspectiveCamera key={`camera-${viewKey}`} makeDefault position={[1.35, 1.05, 1.8]} fov={42} />
            <ambientLight intensity={1.35} />
            <directionalLight position={[2, 4, 3]} intensity={2.2} castShadow />
            <Grid
              position={[0, -0.005, 0]}
              args={[2.4, 2.4]}
              cellSize={0.1}
              sectionSize={0.5}
              cellColor="#ddd4e8"
              sectionColor="#9b82b7"
              fadeDistance={4}
            />
            {points && <Skeleton points={points} connections={data.connections} />}
            <OrbitControls
              key={`controls-${viewKey}`}
              target={[0, 0.75, 0]}
              minDistance={0.7}
              maxDistance={4}
              enablePan
              makeDefault
            />
          </Canvas>
          {!points && <div className="viewer-status overlay">Pose unavailable at this moment</div>}
          <button className="reset-view" type="button" onClick={() => setViewKey((value) => value + 1)}>
            Reset view
          </button>
        </>
      )}
    </div>
  );
}
