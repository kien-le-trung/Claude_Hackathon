"use client";

import Link from "next/link";
import dynamic from "next/dynamic";
import { useParams, useRouter } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";

const InteractiveReconstruction = dynamic(
  () => import("./InteractiveReconstruction"),
  { ssr: false },
);

type ErrorType = "bad_back" | "bad_heel";
type ErrorEvent = {
  id: string;
  type: ErrorType;
  start_timestamp_ms: number;
  end_timestamp_ms: number;
  evidence_frame_count: number;
  confidence: number;
  frame_image_url: string | null;
  measurements: Record<string, number | null>;
};
type Analysis = {
  id: string;
  status: "queued" | "processing" | "completed" | "failed";
  error_message?: string | null;
  extraction?: {
    model: string;
    sampling_fps: number;
    detection_rate: number;
    video: { duration_seconds: number; width: number; height: number };
  } | null;
  classification?: {
    result: "good" | "errors_detected" | "insufficient_pose_data";
    event_count: number;
    event_counts: Record<string, number>;
    events: ErrorEvent[];
    classified_frame_count: number;
    classified_frame_coverage: number;
  } | null;
  media?: {
    source_video_url: string | null;
    reconstruction_video_url: string | null;
    skeleton_data_url: string | null;
    reconstruction_fps: number | null;
    smoothing?: {
      method: string;
      window_length: number;
      polynomial_order: number;
    } | null;
    warnings: string[];
  };
};

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const GUIDANCE = {
  bad_back: {
    label: "Back position",
    short: "Back",
    summary: "The model saw a sustained back position that differs from the stable form patterns it learned. This can happen when the torso loses a braced position during the squat.",
    cues: ["Your chest or pelvis shifts abruptly.", "Your spine appears to round or overextend."],
    causes: ["Losing abdominal tension as the squat gets deeper.", "Using a depth or load that is difficult to control."],
    corrections: ["Brace your trunk before each repetition.", "Use a slower tempo with your ribs stacked over your pelvis.", "Reduce depth or load until your torso stays stable."],
  },
  bad_heel: {
    label: "Heel position",
    short: "Heel",
    summary: "The model saw a sustained heel or foot position that differs from stable squat patterns. Pressure may be shifting away from a balanced, planted foot.",
    cues: ["A heel appears to lift or pressure moves toward the toes.", "The foot rocks as you reach the bottom."],
    causes: ["Limited ankle mobility or an unsuitable stance.", "Moving too quickly or shifting balance forward."],
    corrections: ["Keep pressure across three points of each foot.", "Try a slightly wider stance with knees tracking over toes.", "Use a controlled tempo and work on ankle mobility."],
  },
} satisfies Record<ErrorType, {
  label: string; short: string; summary: string;
  cues: string[]; causes: string[]; corrections: string[];
}>;

function overall(count: number) {
  if (count === 0) return { label: "Great form", description: "No sustained form issues were detected in this video.", tone: "great", grade: "A" };
  if (count <= 2) return { label: "A few things to improve", description: "Your squat is looking solid overall. Focus on the cues below.", tone: "watch", grade: "B" };
  return { label: "Needs focused improvement", description: "Several sustained issues appeared. Work on one cue at a time.", tone: "focus", grade: "C" };
}
const measurementLabel = (name: string) =>
  name.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());

const formatTime = (seconds: number) => {
  if (!Number.isFinite(seconds)) return "0:00";
  return `${Math.floor(seconds / 60)}:${Math.floor(seconds % 60)
    .toString()
    .padStart(2, "0")}`;
};

function SynchronizedVideos({ media }: { media: NonNullable<Analysis["media"]> }) {
  const sourceRef = useRef<HTMLVideoElement>(null);
  const reconstructionRef = useRef<HTMLVideoElement>(null);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const sourceUrl = media.source_video_url ? `${API_URL}${media.source_video_url}` : null;
  const reconstructionUrl = media.reconstruction_video_url
    ? `${API_URL}${media.reconstruction_video_url}`
    : null;

  useEffect(() => {
    setPlaying(false);
    setCurrentTime(0);
    setDuration(0);
    return () => {
      sourceRef.current?.pause();
      reconstructionRef.current?.pause();
    };
  }, [sourceUrl, reconstructionUrl]);

  const master = () => sourceRef.current ?? reconstructionRef.current;

  async function togglePlayback() {
    const primary = master();
    if (!primary) return;
    if (!primary.paused) {
      sourceRef.current?.pause();
      reconstructionRef.current?.pause();
      setPlaying(false);
      return;
    }
    if (primary.ended) {
      primary.currentTime = 0;
      if (reconstructionRef.current) reconstructionRef.current.currentTime = 0;
    }
    const players = [sourceRef.current, reconstructionRef.current].filter(
      (player): player is HTMLVideoElement => Boolean(player)
    );
    try {
      await Promise.all(players.map((player) => player.play()));
      setPlaying(true);
    } catch {
      players.forEach((player) => player.pause());
      setPlaying(false);
    }
  }

  function synchronize() {
    const primary = master();
    if (!primary) return;
    setCurrentTime(primary.currentTime);
    const reconstruction = reconstructionRef.current;
    if (
      sourceRef.current &&
      reconstruction &&
      Math.abs(reconstruction.currentTime - primary.currentTime) > 0.1
    ) {
      reconstruction.currentTime = primary.currentTime;
    }
  }

  function seek(value: number) {
    setCurrentTime(value);
    if (sourceRef.current) sourceRef.current.currentTime = value;
    if (reconstructionRef.current) reconstructionRef.current.currentTime = value;
  }

  return (
    <section className="movement-section" aria-labelledby="movement-heading">
      <div className="section-heading">
        <div><p className="eyebrow">Movement reconstruction</p><h3 id="movement-heading">Your squat in motion</h3></div>
        <p>Compare the uploaded movement with its unsmoothed 3D reconstruction.</p>
      </div>
      <div className={`video-comparison ${sourceUrl && reconstructionUrl ? "" : "single"}`}>
        {sourceUrl && (
          <article className="video-panel">
            <div className="video-label"><strong>Uploaded video</strong><span>Normalized for playback</span></div>
            <video
              ref={sourceRef}
              src={sourceUrl}
              playsInline
              preload="metadata"
              onLoadedMetadata={(event) => setDuration(event.currentTarget.duration)}
              onTimeUpdate={synchronize}
              onEnded={() => {
                reconstructionRef.current?.pause();
                setPlaying(false);
              }}
            />
          </article>
        )}
        {reconstructionUrl && (
          <article className="video-panel">
            <div className="video-label"><strong>3D skeleton</strong><span>{media.reconstruction_fps ?? 10} FPS · unsmoothed</span></div>
            <video
              ref={reconstructionRef}
              src={reconstructionUrl}
              playsInline
              muted
              preload="metadata"
              onLoadedMetadata={(event) => {
                if (!sourceUrl) setDuration(event.currentTarget.duration);
              }}
              onTimeUpdate={sourceUrl ? undefined : synchronize}
              onEnded={sourceUrl ? undefined : () => setPlaying(false)}
            />
          </article>
        )}
      </div>
      {(sourceUrl || reconstructionUrl) && (
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
      )}
      {media.warnings.length > 0 && (
        <div className="media-warnings" role="status">
          {media.warnings.map((warning) => <p key={warning}>{warning}</p>)}
        </div>
      )}
    </section>
  );
}

export default function AnalysisPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    async function load() {
      try {
        const response = await fetch(`${API_URL}/api/videos/${id}`, { cache: "no-store" });
        const payload: Analysis & { detail?: string } = await response.json();
        if (!response.ok) throw new Error(payload.detail ?? "Could not load this analysis");
        if (payload.status !== "completed") {
          throw new Error(payload.error_message ?? "This analysis is not complete");
        }
        if (active) setAnalysis(payload);
      } catch (requestError) {
        if (active) setError(requestError instanceof Error ? requestError.message : "Could not load this analysis");
      }
    }
    load();
    return () => { active = false; };
  }, [id]);

  const groups = useMemo(() => {
    const events = analysis?.classification?.events ?? [];
    return (["bad_back", "bad_heel"] as ErrorType[])
      .map((type) => ({ type, events: events.filter((event) => event.type === type) }))
      .filter((group) => group.events.length);
  }, [analysis]);

  async function deleteAnalysis() {
    if (!analysis) return;
    const response = await fetch(`${API_URL}/api/videos/${analysis.id}`, { method: "DELETE" });
    if (!response.ok) {
      setError("Could not delete this analysis");
      return;
    }
    router.push("/");
  }

  if (error) {
    return (
      <main className="report-main">
        <header className="site-header"><Link className="brand" href="/"><span className="brand-mark">S</span>SquatSpot</Link></header>
        <section className="report-state"><span className="result-icon">!</span><h1>Analysis unavailable</h1><p>{error}</p><Link className="primary-link" href="/">Analyze another video</Link></section>
      </main>
    );
  }
  if (!analysis?.extraction || !analysis.classification) {
    return (
      <main className="report-main">
        <header className="site-header"><Link className="brand" href="/"><span className="brand-mark">S</span>SquatSpot</Link></header>
        <section className="report-state" aria-live="polite"><span className="loading-ring" /><h1>Loading your report</h1><p>Retrieving analysis details…</p></section>
      </main>
    );
  }

  const classification = analysis.classification;
  const extraction = analysis.extraction;
  const score = overall(classification.event_count);
  const averageConfidence = classification.events.length
    ? classification.events.reduce((sum, event) => sum + event.confidence, 0) / classification.events.length
    : null;

  return (
    <main className="report-main">
      <header className="site-header">
        <Link className="brand" href="/"><span className="brand-mark">S</span>SquatSpot</Link>
        <Link className="back-link" href="/">← Analyze another video</Link>
      </header>
      <section className="report-heading">
        <p className="eyebrow">Your analysis</p>
        <h1>Squat form report</h1>
        <span className="completed-pill">✓ Analysis complete</span>
      </section>

      <section className="results-shell report-results">
        {classification.result === "insufficient_pose_data" ? (
          <section className="insufficient-panel">
            <span className="result-icon">?</span>
            <div><h3>Not enough pose data to assess form</h3><p>Try again with a clear full-body side view, brighter lighting, and your hips, knees, ankles, and feet visible.</p></div>
          </section>
        ) : (
          <>
            <section className={`overall-panel ${score.tone}`}>
              <div className="grade-ring" aria-label={`Overall grade ${score.grade}`}><strong>{score.grade}</strong><span>Overall</span></div>
              <div className="overall-copy"><p className="overline">Overall analysis</p><h3>{score.label}</h3><p>{score.description}</p></div>
              <div className="overall-count"><strong>{classification.event_count}</strong><span>sustained {classification.event_count === 1 ? "issue" : "issues"}</span></div>
            </section>
            <div className="stat-grid">
              <article><span>Total issues</span><strong>{classification.event_count}</strong><small>Across the full video</small></article>
              <article><span>Back position</span><strong>{classification.event_counts.bad_back ?? 0}</strong><small>Detected events</small></article>
              <article><span>Heel position</span><strong>{classification.event_counts.bad_heel ?? 0}</strong><small>Detected events</small></article>
              <article><span>Pose coverage</span><strong>{classification.classified_frame_coverage}%</strong><small>{classification.classified_frame_count} frames classified</small></article>
              <article><span>Avg. confidence</span><strong>{averageConfidence === null ? "—" : `${(averageConfidence * 100).toFixed(0)}%`}</strong><small>Model certainty, not severity</small></article>
            </div>

            {analysis.media && analysis.media.skeleton_data_url
              ? <InteractiveReconstruction media={analysis.media} apiUrl={API_URL} />
              : analysis.media && <SynchronizedVideos media={analysis.media} />}

            {groups.length ? (
              <section className="feedback-section">
                <div className="section-heading"><div><p className="eyebrow">Focused feedback</p><h3>What we noticed</h3></div><p>Open each section for an explanation and coaching cues.</p></div>
                <div className="feedback-list">
                  {groups.map(({ type, events }) => {
                    const guide = GUIDANCE[type];
                    return (
                      <details className="feedback-group" key={type} open>
                        <summary><span className={`issue-symbol ${type}`}>{type === "bad_back" ? "↗" : "⌁"}</span><span><strong>{guide.label}</strong><small>{events.length} sustained {events.length === 1 ? "event" : "events"} detected</small></span><span className="chevron">⌄</span></summary>
                        <div className="feedback-content">
                          <p className="guidance-summary">{guide.summary}</p>
                          <div className="guidance-grid">
                            <div><h4>What to look for</h4><ul>{guide.cues.map((item) => <li key={item}>{item}</li>)}</ul></div>
                            <div><h4>Common causes</h4><ul>{guide.causes.map((item) => <li key={item}>{item}</li>)}</ul></div>
                            <div className="correction-box"><h4>Try this next</h4><ul>{guide.corrections.map((item) => <li key={item}>{item}</li>)}</ul></div>
                          </div>
                          <div className="event-list">
                            {events.map((event, index) => (
                              <article className="event-card" key={event.id}>
                                {event.frame_image_url ? <img alt={`Representative ${guide.label.toLowerCase()} frame`} src={`${API_URL}${event.frame_image_url}`} /> : <div className="image-fallback">Representative frame unavailable</div>}
                                <div className="event-body">
                                  <div className="event-title"><h4>{guide.short} event {index + 1}</h4><span>{(event.confidence * 100).toFixed(0)}% confidence</span></div>
                                  <p>{(event.start_timestamp_ms / 1000).toFixed(1)}–{(event.end_timestamp_ms / 1000).toFixed(1)} sec · {event.evidence_frame_count} evidence frames</p>
                                  <details className="technical-details"><summary>View technical measurements</summary><dl className="measurements">
                                    {Object.entries(event.measurements).filter((entry): entry is [string, number] => Number.isFinite(entry[1])).slice(0, 6).map(([name, value]) => <div key={name}><dt>{measurementLabel(name)}</dt><dd>{value.toFixed(1)}</dd></div>)}
                                  </dl></details>
                                </div>
                              </article>
                            ))}
                          </div>
                        </div>
                      </details>
                    );
                  })}
                </div>
              </section>
            ) : <section className="success-panel"><span>✓</span><div><h3>No sustained issues detected</h3><p>Keep practicing the controlled, balanced movement shown in this video.</p></div></section>}
          </>
        )}

        <details className="analysis-details">
          <summary>Analysis and video details</summary>
          <dl className="metadata-grid">
            <div><dt>Pose detection</dt><dd>{extraction.detection_rate}%</dd></div>
            <div><dt>Sampling rate</dt><dd>{extraction.sampling_fps} FPS</dd></div>
            <div><dt>Duration</dt><dd>{extraction.video.duration_seconds.toFixed(1)} sec</dd></div>
            <div><dt>Resolution</dt><dd>{extraction.video.width} × {extraction.video.height}</dd></div>
            <div><dt>Model</dt><dd>{extraction.model}</dd></div>
            <div><dt>Status</dt><dd>Completed</dd></div>
          </dl>
        </details>
        <div className="result-footer">
          <p><strong>Keep this in context.</strong> Confidence reflects model certainty, not severity. SquatSpot is a lightweight coaching aid and does not provide medical advice.</p>
          <button className="delete-button" onClick={deleteAnalysis} type="button">Delete analysis</button>
        </div>
      </section>
      <footer><strong>SquatSpot</strong><span>Move better, one rep at a time.</span></footer>
    </main>
  );
}
