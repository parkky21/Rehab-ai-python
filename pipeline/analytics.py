"""
Session Analytics Dashboard
============================

Generates a multi-panel matplotlib figure with deep analysis after each session:
- Per-rep score breakdown (bar chart)
- Score component distribution (histograms)
- Score trend line across reps
- Radar chart of average component scores
- Constructive text feedback summary

Called from app.py when the user clicks "Stop Analysis".
"""

import matplotlib
matplotlib.use("TkAgg")  # Use Tk backend since we're in a Tkinter app

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pipeline.session import Session


# Color palette
COLORS = {
    "rom": "#3498DB",
    "stability": "#2ECC71",
    "tempo": "#E67E22",
    "asymmetry": "#9B59B6",
    "final": "#E74C3C",
    "background": "#1a1a2e",
    "card": "#16213e",
    "text": "#e0e0e0",
    "grid": "#333355",
}


def generate_feedback_text(session: Session) -> list[str]:
    """Generate constructive feedback based on session data."""
    feedback = []
    summary = session.summary()

    avg_final = summary["avg_final_score"]
    avg_rom = summary["avg_rom_score"]
    avg_stability = summary["avg_stability_score"]
    avg_tempo = summary["avg_tempo_score"]

    # Overall performance
    if avg_final >= 85:
        feedback.append("🏆 Excellent session! Your form is consistently strong.")
    elif avg_final >= 70:
        feedback.append("👍 Good session! Some room for improvement.")
    elif avg_final >= 50:
        feedback.append("💪 Keep practicing! Focus on the weaker areas below.")
    else:
        feedback.append("🎯 This is a starting point — every rep builds strength.")

    # ROM-specific
    if avg_rom < 60:
        feedback.append("📐 ROM: Try to increase your range of motion gradually.")
    elif avg_rom >= 90:
        feedback.append("📐 ROM: Great range of motion — full movement achieved!")

    # Stability-specific
    if avg_stability < 60:
        feedback.append("⚖️ Stability: Focus on keeping your body still during the exercise.")
    elif avg_stability >= 90:
        feedback.append("⚖️ Stability: Excellent balance and control!")

    # Tempo-specific
    if avg_tempo < 60:
        feedback.append("⏱️ Tempo: Try to maintain a slow, controlled pace (~4 seconds per rep).")
    elif avg_tempo >= 90:
        feedback.append("⏱️ Tempo: Perfect controlled rhythm!")

    # Consistency
    if len(session.reps) >= 3:
        scores = [r.final_score for r in session.reps]
        std = np.std(scores)
        if std < 5:
            feedback.append("📊 Consistency: Very consistent performance across all reps!")
        elif std > 15:
            feedback.append("📊 Consistency: Try to maintain more even quality across reps.")

    return feedback


def show_session_analytics(session: Session):
    """
    Open a matplotlib window with deep session analysis.
    Generates 4 charts + constructive feedback text.
    """
    if session.total_reps == 0:
        return

    # Extract data
    reps = session.reps
    rep_nums = [r.rep_number for r in reps]
    rom_scores = [r.rom_score for r in reps]
    stability_scores = [r.stability_score for r in reps]
    tempo_scores = [r.tempo_score for r in reps]
    final_scores = [r.final_score for r in reps]
    feedback_text = generate_feedback_text(session)

    # Create figure with dark theme
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(COLORS["background"])
    fig.suptitle(
        f"Session Analysis — {session.exercise_name}",
        fontsize=18, fontweight="bold", color="white", y=0.98,
    )

    # Grid: 2 rows, 3 columns (last column = feedback text)
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3,
                          left=0.06, right=0.96, top=0.92, bottom=0.06)

    # ======== 1. Per-Rep Score Breakdown (Stacked Bar) ========
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(COLORS["card"])
    x = np.arange(len(rep_nums))
    width = 0.6

    ax1.bar(x, rom_scores, width, label="ROM", color=COLORS["rom"], alpha=0.9)
    ax1.bar(x, stability_scores, width, bottom=rom_scores, label="Stability",
            color=COLORS["stability"], alpha=0.9)
    bottom2 = np.array(rom_scores) + np.array(stability_scores)
    ax1.bar(x, tempo_scores, width, bottom=bottom2, label="Tempo",
            color=COLORS["tempo"], alpha=0.9)

    ax1.set_xlabel("Rep #", color=COLORS["text"])
    ax1.set_ylabel("Score", color=COLORS["text"])
    ax1.set_title("Per-Rep Score Breakdown", fontweight="bold", color="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels(rep_nums)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.tick_params(colors=COLORS["text"])

    # ======== 2. Final Score Trend Line ========
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(COLORS["card"])
    ax2.plot(rep_nums, final_scores, "o-", color=COLORS["final"], linewidth=2.5,
             markersize=8, markerfacecolor="white", markeredgecolor=COLORS["final"])

    # Add score labels on points
    for i, score in enumerate(final_scores):
        ax2.annotate(f"{score:.0f}", (rep_nums[i], score),
                     textcoords="offset points", xytext=(0, 10),
                     ha="center", fontsize=9, color="white", fontweight="bold")

    # Average line
    avg_final = np.mean(final_scores)
    ax2.axhline(y=avg_final, color="#FFD700", linestyle="--", linewidth=1.5, alpha=0.7,
                label=f"Avg: {avg_final:.1f}")

    ax2.set_xlabel("Rep #", color=COLORS["text"])
    ax2.set_ylabel("Final Score", color=COLORS["text"])
    ax2.set_title("Score Trend", fontweight="bold", color="white")
    ax2.set_ylim(0, 110)
    ax2.legend(fontsize=9)
    ax2.tick_params(colors=COLORS["text"])

    # ======== 3. Score Distribution Histogram ========
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(COLORS["card"])

    bins = np.linspace(0, 100, 11)
    ax3.hist(rom_scores, bins=bins, alpha=0.7, label="ROM", color=COLORS["rom"], edgecolor="white", linewidth=0.5)
    ax3.hist(stability_scores, bins=bins, alpha=0.6, label="Stability", color=COLORS["stability"], edgecolor="white", linewidth=0.5)
    ax3.hist(tempo_scores, bins=bins, alpha=0.5, label="Tempo", color=COLORS["tempo"], edgecolor="white", linewidth=0.5)

    ax3.set_xlabel("Score Range", color=COLORS["text"])
    ax3.set_ylabel("Frequency", color=COLORS["text"])
    ax3.set_title("Score Distribution", fontweight="bold", color="white")
    ax3.legend(fontsize=8)
    ax3.tick_params(colors=COLORS["text"])

    # ======== 4. Radar Chart (Average Scores) ========
    ax4 = fig.add_subplot(gs[1, 1], polar=True)
    ax4.set_facecolor(COLORS["card"])

    categories = ["ROM", "Stability", "Tempo", "Final"]
    values = [
        np.mean(rom_scores),
        np.mean(stability_scores),
        np.mean(tempo_scores),
        avg_final,
    ]
    values += values[:1]  # Close the polygon

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    ax4.fill(angles, values, color=COLORS["final"], alpha=0.25)
    ax4.plot(angles, values, "o-", color=COLORS["final"], linewidth=2)
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(categories, color="white", fontweight="bold", fontsize=10)
    ax4.set_ylim(0, 100)
    ax4.set_title("Average Score Profile", fontweight="bold", color="white", pad=20)
    ax4.tick_params(colors=COLORS["text"])
    # Add value labels on radar
    for angle, value in zip(angles[:-1], values[:-1]):
        ax4.annotate(f"{value:.0f}", xy=(angle, value),
                     fontsize=10, ha="center", color="white", fontweight="bold")

    # ======== 5. Feedback Text Panel ========
    ax5 = fig.add_subplot(gs[:, 2])
    ax5.set_facecolor(COLORS["card"])
    ax5.axis("off")

    summary = session.summary()
    header_text = (
        f"SESSION SUMMARY\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Exercise:  {session.exercise_name}\n"
        f"Total Reps:  {summary['total_reps']}\n"
        f"Duration:  {summary['duration_seconds']:.0f}s\n"
        f"━━━━━━━━━━━━━━━━━━━━\n\n"
        f"AVG SCORES\n"
        f"  Final:      {summary['avg_final_score']:.1f} / 100\n"
        f"  ROM:        {summary['avg_rom_score']:.1f} / 100\n"
        f"  Stability:  {summary['avg_stability_score']:.1f} / 100\n"
        f"  Tempo:      {summary['avg_tempo_score']:.1f} / 100\n"
        f"━━━━━━━━━━━━━━━━━━━━\n\n"
        f"FEEDBACK\n"
    )

    for fb in feedback_text:
        header_text += f"  {fb}\n"

    ax5.text(
        0.05, 0.95, header_text,
        transform=ax5.transAxes,
        fontsize=11, color=COLORS["text"],
        verticalalignment="top",
        fontfamily="monospace",
        linespacing=1.6,
    )

    plt.show(block=False)
