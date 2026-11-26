import React, { useState } from "react";

function ScoreBar({ label, value, colorClass }) {
  const pct = Math.max(0, Math.min(100, Math.round((value || 0) * 100)));
  return (
    <div className="score-bar-row">
      <span className="score-bar-label">{label}</span>
      <div className="score-bar-track">
        <div
          className={`score-bar-fill ${colorClass}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="score-bar-val">{(value || 0).toFixed(3)}</span>
    </div>
  );
}

export default function RecommendationList({
  responseMeta,
  recommendations,
  selectedRec,
  onSelectRec,
}) {
  const [modalRec, setModalRec] = useState(null);

  return (
    <div className="rec-panel">
      <div className="rec-header">
        <h2>Top-K Recommendations</h2>
        {responseMeta && (
          <div className="meta-badges">
            <span className="badge badge-model">{responseMeta.model_used}</span>
            <span className="badge badge-time">
              {responseMeta.inference_time_ms} ms
            </span>
            {responseMeta.is_cold_start && (
              <span className="badge badge-cold">Cold-Start</span>
            )}
          </div>
        )}
      </div>

      {recommendations.length === 0 ? (
        <div className="empty-state">
          No venues match the current radius or category constraints. Try
          widening the radius or clearing category filters.
        </div>
      ) : (
        <div className="rec-cards">
          {recommendations.map((rec) => {
            const isSelected = selectedRec && selectedRec.poi_id === rec.poi_id;
            return (
              <div
                key={`${rec.poi_id}-${rec.rank}`}
                className={isSelected ? "rec-card selected" : "rec-card"}
                role="button"
                tabIndex={0}
                onClick={() => onSelectRec(rec)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    onSelectRec(rec);
                  }
                }}
              >
                <div className="rec-card-top">
                  <span className="rec-rank">#{rec.rank}</span>
                  <div className="rec-title-wrap">
                    <div className="rec-title">{rec.name}</div>
                    <div className="rec-sub">
                      <span className="cat-pill">{rec.category}</span>
                      <span>{rec.distance_km.toFixed(2)} km</span>
                      <span>Score {rec.total_score.toFixed(3)}</span>
                    </div>
                  </div>
                  <button
                    type="button"
                    className="btn-score-popup"
                    onClick={(e) => {
                      e.stopPropagation();
                      setModalRec(rec);
                    }}
                    title="View recommender score breakdown"
                  >
                    Scores
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {modalRec && (
        <div
          className="score-modal-backdrop"
          onClick={() => setModalRec(null)}
          role="presentation"
        >
          <div
            className="score-modal"
            onClick={(e) => e.stopPropagation()}
            role="dialog"
            aria-modal="true"
          >
            <div className="score-modal-header">
              <div>
                <div className="score-modal-title">
                  #{modalRec.rank} {modalRec.name}
                </div>
                <div className="score-modal-sub">
                  {modalRec.category} | {modalRec.distance_km.toFixed(2)} km | Total Score{" "}
                  {modalRec.total_score.toFixed(4)}
                </div>
              </div>
              <button
                type="button"
                className="modal-close-btn"
                onClick={() => setModalRec(null)}
                aria-label="Close popup"
              >
                x
              </button>
            </div>

            <div className="score-modal-exp">{modalRec.explanation}</div>

            <div className="score-modal-bars">
              <ScoreBar
                label="Sequential"
                value={modalRec.score_breakdown.sequential}
                colorClass="fill-seq"
              />
              <ScoreBar
                label="Content"
                value={modalRec.score_breakdown.content}
                colorClass="fill-cnt"
              />
              <ScoreBar
                label="CF-BPR"
                value={modalRec.score_breakdown.cf}
                colorClass="fill-cf"
              />
              <ScoreBar
                label="Spatial"
                value={modalRec.score_breakdown.spatial}
                colorClass="fill-geo"
              />
            </div>

            <div className="score-modal-footer">
              <button
                type="button"
                className="btn-mini"
                onClick={() => {
                  onSelectRec(modalRec);
                  setModalRec(null);
                }}
              >
                Focus on Map
              </button>
              <button
                type="button"
                className="btn-mini btn-mini-primary"
                onClick={() => setModalRec(null)}
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
