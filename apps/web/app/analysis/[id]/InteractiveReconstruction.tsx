"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import SkeletonViewer from "./SkeletonViewer";

type Media = {
  source_video_url: string | null;
  reconstruction_video_url: string | null;
  skeleton_data_url: string | null;
  reconstruction_fps: number | null;
  warnings: string[];
};

const formatTime = (seconds: number) => {
  if (!Number.isFinite(seconds)) return "0:00";
  return `${Math.floor(seconds / 60)}:${Math.floor(seconds % 60)
    .toString()
    .padStart(2, "0")}`;
};

export default function InteractiveReconstruction({
  media,
  apiUrl,
}: {
  media: Media;
  apiUrl: string;
}) {
  const sourceRef = useRef<HTMLVideoElement>(null);
  const clockRef = useRef({ wallTime: 0, mediaTime: 0 });
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const sourceUrl = media.source_video_url ? `${apiUrl}${media.source_video_url}` : null;
  const skeletonUrl = `${apiUrl}${media.skeleton_data_url}`;

  useEffect(() => {
    setPlaying(false);
    setCurrentTime(0);
    setDuration(0);
    return () => sourceRef.current?.pause();
  }, [sourceUrl, skeletonUrl]);

  useEffect(() => {
    if (!playing) return;
    let request = 0;
    const tick = (wallTime: number) => {
      if (sourceRef.current) {
        setCurrentTime(sourceRef.current.currentTime);
      } else {
        const elapsed = (wallTime - clockRef.current.wallTime) / 1000;
        const next = clockRef.current.mediaTime + elapsed;
        if (duration > 0 && next >= duration) {
          setCurrentTime(duration);
          setPlaying(false);
          return;
        }
        setCurrentTime(next);
      }
      request = requestAnimationFrame(tick);
    };
    request = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(request);
  }, [playing, duration]);

  async function togglePlayback() {
    const source = sourceRef.current;
    if (!source) {
      if (playing) {
        setPlaying(false);
        return;
      }
      const start = duration > 0 && currentTime >= duration ? 0 : currentTime;
      setCurrentTime(start);
      clockRef.current = { wallTime: performance.now(), mediaTime: start };
      setPlaying(true);
      return;
    }
    if (!source.paused) {
      source.pause();
      setPlaying(false);
      return;
    }
    if (source.ended) source.currentTime = 0;
    try {
      await source.play();
      setPlaying(true);
    } catch {
      source.pause();
      setPlaying(false);
    }
  }

  function seek(value: number) {
    setCurrentTime(value);
    clockRef.current = { wallTime: performance.now(), mediaTime: value };
    if (sourceRef.current) sourceRef.current.currentTime = value;
  }

  const skeletonLoaded = useCallback((skeletonDuration: number) => {
    if (!sourceUrl) setDuration(skeletonDuration);
  }, [sourceUrl]);

  return (
    <section className="movement-section" aria-labelledby="movement-heading">
      <div className="section-heading">
        <div><p className="eyebrow">Movement reconstruction</p><h3 id="movement-heading">Your squat in motion</h3></div>
        <p>Compare the uploaded movement with its smoothed, interactive 3D reconstruction.</p>
      </div>
      <div className={`video-comparison ${sourceUrl ? "" : "single"}`}>
        {sourceUrl && (
          <article className="video-panel">
            <div className="video-label"><strong>Uploaded video</strong><span>Normalized for playback</span></div>
            <video
              ref={sourceRef}
              src={sourceUrl}
              playsInline
              preload="metadata"
              onLoadedMetadata={(event) => setDuration(event.currentTarget.duration)}
              onEnded={() => setPlaying(false)}
            />
          </article>
        )}
        <article className="video-panel">
          <div className="video-label">
            <strong>3D skeleton</strong>
            <span>Interactive 3D · Savitzky–Golay smoothed</span>
          </div>
          <SkeletonViewer
            url={skeletonUrl}
            currentTime={currentTime}
            onLoaded={skeletonLoaded}
          />
        </article>
      </div>
      <div className="shared-video-controls">
        <button type="button" onClick={togglePlayback}>{playing ? "Pause" : "Play"}</button>
        <span>{formatTime(currentTime)}</span>
        <input
          aria-label="Video position"
          type="range"
          min="0"
          max={duration || 0}
          step="0.01"
          value={Math.min(currentTime, duration || 0)}
          onChange={(event) => seek(Number(event.target.value))}
        />
        <span>{formatTime(duration)}</span>
      </div>
      {media.warnings.length > 0 && (
        <div className="media-warnings" role="status">
          {media.warnings.map((warning) => <p key={warning}>{warning}</p>)}
        </div>
      )}
    </section>
  );
}
