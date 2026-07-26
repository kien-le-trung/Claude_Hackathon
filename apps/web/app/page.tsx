"use client";

import { FormEvent, useMemo, useState } from "react";
import { useRouter } from "next/navigation";

type ErrorType = "bad_back" | "bad_heel";

type ErrorEvent = {
  id: string;
  type: ErrorType;
  start_timestamp_ms: number;
  end_timestamp_ms: number;
  duration_ms: number;
  evidence_frame_count: number;
  confidence: number;
  representative_frame_number: number;
  representative_timestamp_ms: number;
  frame_image_url: string | null;
  measurements: Record<string, number | null>;
  artifact_warning?: string | null;
};

type AnalysisResponse = {
  id: string;
  status: "queued" | "processing" | "completed" | "failed";
  extraction?: {
    schema_version: number;
    model: string;
    video: {
      fps: number;
      total_frames: number;
      duration_seconds: number;
      width: number;
      height: number;
      content_type: string;
    };
    sampling_fps: number;
    sampled_frame_count: number;
    detected_frame_count: number;
    detection_rate: number;
  } | null;
  classification?: {
    result: "good" | "errors_detected" | "insufficient_pose_data";
    event_count: number;
    event_counts: Record<string, number>;
    events: ErrorEvent[];
    classified_frame_count: number;
    sampled_frame_count: number;
    classified_frame_coverage: number;
  } | null;
  progress?: {
    stage?: "extracting" | "classifying" | "preparing_video" | "rendering_reconstruction" | "completed";
    decoded_frame_count: number;
    sampled_frame_count: number;
    detected_frame_count: number;
    classified_frame_count: number;
    total_frame_count: number;
    percent: number;
  } | null;
  error_message?: string | null;
};

const ERROR_GUIDANCE: Record<
  ErrorType,
  {
    label: string;
    shortLabel: string;
    summary: string;
    cues: string[];
    causes: string[];
    corrections: string[];
  }
> = {
  bad_back: {
    label: "Back position",
    shortLabel: "Back",
    summary:
      "The model saw a sustained back position that differs from the form patterns it learned. This can happen when the torso loses a stable, braced position during the squat.",
    cues: [
      "Your chest or pelvis shifts abruptly during the descent or ascent.",
      "Your spine appears to round or overextend instead of moving as one stable unit.",
    ],
    causes: [
      "Losing abdominal tension as the squat gets deeper.",
      "Using a depth or load that is difficult to control.",
    ],
    corrections: [
      "Take a breath and brace your trunk before each repetition.",
      "Practice a slower bodyweight squat while keeping your ribs stacked over your pelvis.",
      "Reduce depth or load until you can maintain the same torso position.",
    ],
  },
  bad_heel: {
    label: "Heel position",
    shortLabel: "Heel",
    summary:
      "The model saw a sustained heel or foot position that differs from stable squat patterns. It may indicate that pressure is shifting away from a balanced, planted foot.",
    cues: [
      "A heel appears to lift or foot pressure moves noticeably toward the toes.",
      "The foot rocks or changes position as you reach the bottom of the squat.",
    ],
    causes: [
      "Limited ankle mobility or a stance that does not suit your current range.",
      "Moving too quickly or shifting your balance forward.",
    ],
    corrections: [
      "Keep pressure across the heel, base of the big toe, and base of the little toe.",
      "Try a slightly wider stance and let your knees track in line with your toes.",
      "Use a controlled tempo and work on ankle mobility before adding load.",
    ],
  },
};

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const sleep = (milliseconds: number) =>
  new Promise((resolve) => setTimeout(resolve, milliseconds));
const measurementLabel = (name: string) =>
  name.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());

function overallResult(eventCount: number) {
  if (eventCount === 0) {
    return {
      label: "Great form",
      description: "No sustained form issues were detected in this video.",
      tone: "great",
      grade: "A",
    };
  }
  if (eventCount <= 2) {
    return {
      label: "A few things to improve",
      description: "Your squat is looking solid overall. Focus on the cues below.",
      tone: "watch",
      grade: "B",
    };
  }
  return {
    label: "Needs focused improvement",
    description: "Several sustained issues appeared. Work on one cue at a time.",
    tone: "focus",
    grade: "C",
  };
}

export default function HomePage() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const groupedEvents = useMemo(() => {
    const events = analysis?.classification?.events ?? [];
    return (["bad_back", "bad_heel"] as ErrorType[])
      .map((type) => ({ type, events: events.filter((item) => item.type === type) }))
      .filter((group) => group.events.length > 0);
  }, [analysis]);

  async function poll(videoId: string): Promise<AnalysisResponse> {
    for (;;) {
      const response = await fetch(`${API_URL}/api/videos/${videoId}`, {
        cache: "no-store",
      });
      const payload: AnalysisResponse = await response.json();
      if (!response.ok) throw new Error("Could not retrieve analysis status");
      setAnalysis(payload);
      if (payload.status === "completed") return payload;
      if (payload.status === "failed") {
        throw new Error(payload.error_message ?? "Analysis failed");
      }
      await sleep(1500);
    }
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file) return;
    setIsSubmitting(true);
    setError(null);
    setAnalysis(null);
    const body = new FormData();
    body.append("video", file);

    try {
      const response = await fetch(`${API_URL}/api/videos`, {
        method: "POST",
        body,
      });
      const payload: AnalysisResponse & { detail?: string } = await response.json();
      if (!response.ok) throw new Error(payload.detail ?? "Analysis failed");
      setAnalysis(payload);
      await poll(payload.id);
      router.push(`/analysis/${payload.id}`);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Analysis failed");
    } finally {
      setIsSubmitting(false);
    }
  }

  async function deleteAnalysis() {
    if (!analysis) return;
    const response = await fetch(`${API_URL}/api/videos/${analysis.id}`, {
      method: "DELETE",
    });
    if (!response.ok) {
      setError("Could not delete this analysis");
      return;
    }
    setAnalysis(null);
    setFile(null);
  }

  const classification = analysis?.classification!;
  const result = overallResult(classification?.event_count ?? 0);
  const averageConfidence =
    classification?.events.length
      ? classification.events.reduce((sum, item) => sum + item.confidence, 0) /
        classification.events.length
      : 0;

  return (
    <main>
      <header className="site-header">
        <a className="brand" href="#top" aria-label="SquatSpot home">
          <span className="brand-mark" aria-hidden="true">S</span>
          SquatSpot
        </a>
        <span className="privacy-pill">Private by design</span>
      </header>

      <section className="hero" id="top">
        <div className="hero-copy">
          <p className="eyebrow">AI-powered form feedback</p>
          <h1>See your squat.<br /><span>Move with confidence.</span></h1>
          <p className="lede">
            Upload a short squat video and get focused feedback on your back and
            heel position in moments.
          </p>
          <div className="trust-row" aria-label="Product highlights">
            <span>✓ Local analysis</span>
            <span>✓ Actionable cues</span>
            <span>✓ Media kept only until you delete the analysis</span>
          </div>
        </div>

        <section className="upload-card" aria-labelledby="upload-heading">
          <div className="card-heading">
            <span className="step-number">1</span>
            <div>
              <p className="overline">Start your analysis</p>
              <h2 id="upload-heading">Upload a squat video</h2>
            </div>
          </div>
          <form onSubmit={submit}>
            <label className={`upload ${file ? "has-file" : ""}`}>
              <input
                type="file"
                accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska"
                onChange={(event) => setFile(event.target.files?.[0] ?? null)}
              />
              <span className="upload-icon" aria-hidden="true">↑</span>
              <strong>{file ? "Video ready" : "Choose or drop your video"}</strong>
              <span className="file-name">
                {file ? file.name : "MP4, AVI, MOV, or MKV · up to 100 MB"}
              </span>
              <span className="browse-label">{file ? "Choose another" : "Browse files"}</span>
            </label>
            <button className="primary-button" disabled={!file || isSubmitting} type="submit">
              {isSubmitting ? "Analyzing…" : "Analyze my squat"}
              <span aria-hidden="true">→</span>
            </button>
          </form>

          {analysis && !analysis.extraction && !error && (
            <section className="progress-panel" aria-live="polite">
              <div className="progress-copy">
                <strong>
                  {analysis.progress?.stage === "preparing_video"
                    ? "Preparing your video"
                    : analysis.progress?.stage === "rendering_reconstruction"
                      ? "Rendering your 3D reconstruction"
                      : analysis.progress?.stage === "classifying"
                        ? "Evaluating your form"
                        : "Analyzing your movement"}
                </strong>
                <span>{analysis.progress?.percent.toFixed(0) ?? 0}%</span>
              </div>
              <progress max="100" value={analysis.progress?.percent ?? 0} />
              <p>
                {analysis.progress
                  ? `${analysis.progress.sampled_frame_count} frames sampled · ${analysis.progress.detected_frame_count} poses found`
                  : `Analysis ${analysis.status}`}
              </p>
            </section>
          )}
          {error && <p className="error" role="alert">{error}</p>}
        </section>
      </section>

      <section className="recording-guide" aria-labelledby="recording-heading">
        <div>
          <p className="eyebrow">Before you record</p>
          <h2 id="recording-heading">Help the model see your movement clearly</h2>
        </div>
        <div className="tip-grid">
          <article><span>01</span><h3>Show your full body</h3><p>Keep your hips, knees, ankles, and feet visible for the whole rep.</p></article>
          <article><span>02</span><h3>Use a side view</h3><p>Set your camera around hip height and avoid extreme camera angles.</p></article>
          <article><span>03</span><h3>Keep it short</h3><p>Record 2–5 controlled reps in good lighting, without other people in frame.</p></article>
        </div>
      </section>

      {analysis?.extraction && classification && false && (
        <section className="results-shell" aria-labelledby="results-heading">
          <div className="results-title">
            <div>
              <p className="eyebrow">Your analysis</p>
              <h2 id="results-heading">Squat form report</h2>
            </div>
            <span className="completed-pill">✓ Analysis complete</span>
          </div>

          {classification?.result === "insufficient_pose_data" ? (
            <section className="insufficient-panel">
              <span className="result-icon" aria-hidden="true">?</span>
              <div>
                <h3>Not enough pose data to assess form</h3>
                <p>
                  Too few frames contained reliable biomechanical measurements. Try
                  again with a clear full-body side view, brighter lighting, and your
                  hips, knees, ankles, and feet visible.
                </p>
              </div>
            </section>
          ) : result && classification ? (
            <>
              <section className={`overall-panel ${result.tone}`}>
                <div className="grade-ring" aria-label={`Overall grade ${result.grade}`}>
                  <strong>{result.grade}</strong><span>Overall</span>
                </div>
                <div className="overall-copy">
                  <p className="overline">Overall analysis</p>
                  <h3>{result.label}</h3>
                  <p>{result.description}</p>
                </div>
                <div className="overall-count">
                  <strong>{classification.event_count}</strong>
                  <span>sustained {classification.event_count === 1 ? "issue" : "issues"}</span>
                </div>
              </section>

              <div className="stat-grid">
                <article><span>Total issues</span><strong>{classification.event_count}</strong><small>Across the full video</small></article>
                <article><span>Back position</span><strong>{classification.event_counts.bad_back ?? 0}</strong><small>Detected events</small></article>
                <article><span>Heel position</span><strong>{classification.event_counts.bad_heel ?? 0}</strong><small>Detected events</small></article>
                <article><span>Pose coverage</span><strong>{classification.classified_frame_coverage}%</strong><small>{classification.classified_frame_count} frames classified</small></article>
                <article><span>Avg. confidence</span><strong>{averageConfidence === null ? "—" : `${(averageConfidence * 100).toFixed(0)}%`}</strong><small>Model certainty, not severity</small></article>
              </div>

              {groupedEvents.length > 0 ? (
                <section className="feedback-section">
                  <div className="section-heading">
                    <div><p className="eyebrow">Focused feedback</p><h3>What we noticed</h3></div>
                    <p>Open each section for an explanation and coaching cues.</p>
                  </div>
                  <div className="feedback-list">
                    {groupedEvents.map(({ type, events }) => {
                      const guidance = ERROR_GUIDANCE[type];
                      return (
                        <details className="feedback-group" key={type} open>
                          <summary>
                            <span className={`issue-symbol ${type}`} aria-hidden="true">
                              {type === "bad_back" ? "↗" : "⌁"}
                            </span>
                            <span><strong>{guidance.label}</strong><small>{events.length} sustained {events.length === 1 ? "event" : "events"} detected</small></span>
                            <span className="chevron" aria-hidden="true">⌄</span>
                          </summary>
                          <div className="feedback-content">
                            <p className="guidance-summary">{guidance.summary}</p>
                            <div className="guidance-grid">
                              <div><h4>What to look for</h4><ul>{guidance.cues.map((item) => <li key={item}>{item}</li>)}</ul></div>
                              <div><h4>Common causes</h4><ul>{guidance.causes.map((item) => <li key={item}>{item}</li>)}</ul></div>
                              <div className="correction-box"><h4>Try this next</h4><ul>{guidance.corrections.map((item) => <li key={item}>{item}</li>)}</ul></div>
                            </div>

                            <div className="event-list">
                              {events.map((detectedEvent, index) => (
                                <article className="event-card" key={detectedEvent.id}>
                                  {detectedEvent.frame_image_url ? (
                                    <img
                                      alt={`Representative ${guidance.label.toLowerCase()} frame`}
                                      loading="lazy"
                                      src={`${API_URL}${detectedEvent.frame_image_url}`}
                                    />
                                  ) : (
                                    <div className="image-fallback">Representative frame unavailable</div>
                                  )}
                                  <div className="event-body">
                                    <div className="event-title">
                                      <h4>{guidance.shortLabel} event {index + 1}</h4>
                                      <span>{(detectedEvent.confidence * 100).toFixed(0)}% confidence</span>
                                    </div>
                                    <p>
                                      {(detectedEvent.start_timestamp_ms / 1000).toFixed(1)}–
                                      {(detectedEvent.end_timestamp_ms / 1000).toFixed(1)} sec ·{" "}
                                      {detectedEvent.evidence_frame_count} evidence frames
                                    </p>
                                    <details className="technical-details">
                                      <summary>View technical measurements</summary>
                                      <dl className="measurements">
                                        {Object.entries(detectedEvent.measurements)
                                          .filter((entry): entry is [string, number] => Number.isFinite(entry[1]))
                                          .slice(0, 6)
                                          .map(([name, value]) => (
                                            <div key={name}><dt>{measurementLabel(name)}</dt><dd>{value.toFixed(1)}</dd></div>
                                          ))}
                                      </dl>
                                    </details>
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
              ) : (
                <section className="success-panel">
                  <span aria-hidden="true">✓</span>
                  <div><h3>No sustained issues detected</h3><p>Keep practicing the controlled, balanced movement shown in this video.</p></div>
                </section>
              )}
            </>
          ) : null}

          <details className="analysis-details">
            <summary>Analysis and video details</summary>
            <dl className="metadata-grid">
              <div><dt>Pose detection</dt><dd>{analysis!.extraction!.detection_rate}%</dd></div>
              <div><dt>Sampling rate</dt><dd>{analysis!.extraction!.sampling_fps} FPS</dd></div>
              <div><dt>Duration</dt><dd>{analysis!.extraction!.video.duration_seconds.toFixed(1)} sec</dd></div>
              <div><dt>Resolution</dt><dd>{analysis!.extraction!.video.width} × {analysis!.extraction!.video.height}</dd></div>
              <div><dt>Model</dt><dd>{analysis!.extraction!.model}</dd></div>
              <div><dt>Status</dt><dd>Completed</dd></div>
            </dl>
          </details>

          <div className="result-footer">
            <p>
              <strong>Keep this in context.</strong> Confidence reflects model certainty,
              not the severity of an issue. SquatSpot is a lightweight coaching aid and
              does not provide medical advice.
            </p>
            <button className="delete-button" onClick={deleteAnalysis} type="button">Delete analysis</button>
          </div>
        </section>
      )}

      <section className="how-it-works" aria-labelledby="how-heading">
        <div className="section-heading centered">
          <p className="eyebrow">Simple and private</p>
          <h2 id="how-heading">From video to useful feedback</h2>
        </div>
        <div className="process-grid">
          <article><span>1</span><h3>Upload</h3><p>Select a short video in MP4, AVI, MOV, or MKV format.</p></article>
          <article><span>2</span><h3>Analyze</h3><p>MediaPipe tracks your pose while a local classifier reviews form patterns.</p></article>
          <article><span>3</span><h3>Improve</h3><p>Review representative frames and practice the focused coaching cues.</p></article>
        </div>
        <div className="project-note">
          <span aria-hidden="true">◈</span>
          <div><h3>Built for explainable feedback</h3><p>SquatSpot combines pose landmarks with a lightweight biomechanics classifier. A browser-ready copy and its reconstruction are retained with the analysis so you can review them, then removed when you delete the analysis.</p></div>
        </div>
      </section>

      <footer><strong>SquatSpot</strong><span>Move better, one rep at a time.</span></footer>
    </main>
  );
}
