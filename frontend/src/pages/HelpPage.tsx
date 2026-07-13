/**
 * Help page — the single readable reference for how the pick'em works AND how
 * to use the Discord bot.
 *
 * Renders INSIDE AppShell (which is inside RequireAuth), so this is CONTENT
 * ONLY: no shell, header, nav, or auth guard. Pure static JSX — no hooks, no
 * data fetching, no API/lib imports, no new deps. Topics are native
 * <details>/<summary> accordions styled with the app's theme-aware tokens.
 *
 * The rules content is authoritative (code-derived from scoring.py,
 * pick_validation.py, pick_window.py, odds.py, and PROJECT.md) and the bot
 * content is code-derived from app/bot/*; both are rendered faithfully, not
 * invented or altered — do not over-promise capabilities the bot lacks.
 */
export default function HelpPage() {
  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-bold">Help</h1>
        <p className="mt-1 text-sm text-fg-muted">
          Each NFL week, make your picks against that week's games. Picks are
          graded against the betting line and the final scores, and roll up into
          weekly and season-long standings. Below: the full rules, plus how to
          use the Discord bot.
        </p>
      </header>

      <details
        open
        className="space-y-3 rounded-lg border border-border bg-surface p-4"
      >
        <summary className="cursor-pointer text-lg font-bold text-fg">
          Your weekly picks
        </summary>
        <p className="text-sm text-fg-muted">
          Each week you can make up to five main picks — one of each of the four
          bet types (favorite cover, underdog cover, over, under) plus a Mortal
          Lock wildcard — and one misc pick. Every pick is optional; there's
          no minimum, so make as many or as few as you like.
        </p>
        <ul className="list-disc space-y-1 pl-5 text-sm text-fg-muted">
          <li>
            <span className="font-semibold">Favorite cover</span> — the favorite
            covers the spread.
          </li>
          <li>
            <span className="font-semibold">Underdog cover</span> — the underdog
            covers the spread.
          </li>
          <li>
            <span className="font-semibold">Over</span> — the two teams' combined
            score goes over the total.
          </li>
          <li>
            <span className="font-semibold">Under</span> — the combined score
            stays under the total.
          </li>
        </ul>
        <ul className="list-disc space-y-1 pl-5 text-sm text-fg-muted">
          <li>
            <span className="font-semibold">Mortal Lock</span> — a fifth main pick,
            a wildcard on any game and type of your choosing. A winning Mortal Lock
            is worth more, but a losing one costs you (see Scoring).
          </li>
          <li>
            <span className="font-semibold">Misc</span> — the genuinely different
            one: a free-text pick tied to a game (e.g. "Mahomes throws for
            350+ yards"), graded by an admin rather than automatically.
          </li>
        </ul>
      </details>

      {/* Outcomes and points below are documentation of the backend scoring
          engine — keep in sync with app/services/scoring.py (`_spread_outcome`,
          `_total_outcome`, `_points_for`) if it changes. */}
      <details className="space-y-3 rounded-lg border border-border bg-surface p-4">
        <summary className="cursor-pointer text-lg font-bold text-fg">
          See it in action
        </summary>
        <p className="text-sm text-fg-muted">
          Say the Chiefs are favored by 6.5 over the Broncos, with the total set
          at 44.5. Final: Chiefs 27, Broncos 20. Here's how each pick type grades
          out:
        </p>
        <div className="overflow-x-auto rounded-lg border border-border bg-surface">
          <table className="min-w-full border-collapse text-sm">
            <thead>
              <tr className="border-b border-border text-fg-muted">
                <th className="px-3 py-2 text-left font-semibold">Your pick</th>
                <th className="px-3 py-2 text-left font-semibold">Why</th>
                <th className="px-3 py-2 text-left font-semibold">Result</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-border last:border-0">
                <td className="px-3 py-2 text-left text-fg">
                  Favorite cover (Chiefs −6.5)
                </td>
                <td className="px-3 py-2 text-left text-fg-muted">
                  won by 7 — more than 6.5
                </td>
                <td className="px-3 py-2 text-left tabular-nums text-fg-muted">
                  Win +1
                </td>
              </tr>
              <tr className="border-b border-border last:border-0">
                <td className="px-3 py-2 text-left text-fg">
                  Underdog cover (Broncos +6.5)
                </td>
                <td className="px-3 py-2 text-left text-fg-muted">
                  didn't stay within 6.5
                </td>
                <td className="px-3 py-2 text-left tabular-nums text-fg-muted">
                  Loss 0
                </td>
              </tr>
              <tr className="border-b border-border last:border-0">
                <td className="px-3 py-2 text-left text-fg">Over (44.5)</td>
                <td className="px-3 py-2 text-left text-fg-muted">
                  27 + 20 = 47, over the total
                </td>
                <td className="px-3 py-2 text-left tabular-nums text-fg-muted">
                  Win +1
                </td>
              </tr>
              <tr className="border-b border-border last:border-0">
                <td className="px-3 py-2 text-left text-fg">Under (44.5)</td>
                <td className="px-3 py-2 text-left text-fg-muted">
                  47 isn't under 44.5
                </td>
                <td className="px-3 py-2 text-left tabular-nums text-fg-muted">
                  Loss 0
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <ul className="list-disc space-y-1 pl-5 text-sm text-fg-muted">
          <li>
            <span className="font-semibold">Mortal Lock</span>: if that Favorite
            cover were your lock → +2 (a losing lock costs −1).
          </li>
          <li>
            <span className="font-semibold">Underdog cover</span> doesn't need the
            underdog to win — just to lose by less than the spread. Had the Chiefs
            won 24–20 (by 4), the Broncos would cover → that pick Wins.
          </li>
          <li>
            <span className="font-semibold">Push</span>: on a whole-number line
            (e.g. −3) landing exactly — Chiefs by 3 — both cover picks score 0.
          </li>
          <li>
            <span className="font-semibold">Misc</span> isn't tied to the line —
            an admin grades it.
          </li>
        </ul>
      </details>

      <details className="space-y-3 rounded-lg border border-border bg-surface p-4">
        <summary className="cursor-pointer text-lg font-bold text-fg">
          Roster rules
        </summary>
        <ul className="list-disc space-y-1 pl-5 text-sm text-fg-muted">
          <li>At most one base pick of each of the four types per week.</li>
          <li>The Mortal Lock is the only same-type duplicate allowed.</li>
          <li>
            You can't pick both sides of the same line on the same game (Favorite
            cover AND Underdog cover, or Over AND Under).
          </li>
          <li>A Misc is never a Mortal Lock.</li>
        </ul>
      </details>

      <details className="space-y-3 rounded-lg border border-border bg-surface p-4">
        <summary className="cursor-pointer text-lg font-bold text-fg">
          The pick window
        </summary>
        <ul className="list-disc space-y-1 pl-5 text-sm text-fg-muted">
          <li>
            A week's window OPENS after the previous week's last game ends, and
            CLOSES at that week's first kickoff. (Week 1 is open from the start.)
          </li>
          <li>
            Picking for the WHOLE week ends at that first kickoff — once the
            window closes you can no longer add or change any pick for the week,
            even on games later in the week that haven't kicked off yet.
          </li>
          <li>
            Once the window closes, EVERYONE's picks for that week become
            visible to all players (your own picks are always visible to you).
          </li>
          <li>
            Picks are graded against the locked line (see "When the lines lock"
            below), plus the final scores.
          </li>
        </ul>
      </details>

      <details className="space-y-3 rounded-lg border border-border bg-surface p-4">
        <summary className="cursor-pointer text-lg font-bold text-fg">
          When the lines lock
        </summary>
        <ul className="list-disc space-y-1 pl-5 text-sm text-fg-muted">
          <li>
            Each week's betting line (the spread and the total) can still move
            while it's live — then it LOCKS, and the line you see becomes final.
            Your picks are graded against that locked line, and it won't change
            afterward even if the sportsbooks keep moving it. (You'll also see
            this called the line "freezing.")
          </li>
          <li>
            A week's line locks at noon ET on the Wednesday before that week's
            first kickoff — or at the first kickoff itself, if that somehow comes
            first. That's usually a day or two before picks lock.
          </li>
          <li>
            The bar at the top of the app shows <strong>lines live</strong> while
            the line can still move, and <strong>lines locked</strong> once it's
            final for the week.
          </li>
        </ul>
      </details>

      {/* Point values below are documentation of the backend scoring engine —
          keep in sync with app/services/scoring.py `_points_for` if it changes. */}
      <details className="space-y-3 rounded-lg border border-border bg-surface p-4">
        <summary className="cursor-pointer text-lg font-bold text-fg">
          Scoring
        </summary>
        <div className="overflow-x-auto rounded-lg border border-border bg-surface">
          <table className="min-w-full border-collapse text-sm">
            <thead>
              <tr className="border-b border-border text-fg-muted">
                <th className="px-3 py-2 text-left font-semibold">Outcome</th>
                <th className="px-3 py-2 text-right font-semibold">
                  Normal pick
                </th>
                <th className="px-3 py-2 text-right font-semibold">
                  Mortal Lock
                </th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-border last:border-0">
                <td className="px-3 py-2 text-left text-fg">Win</td>
                <td className="px-3 py-2 text-right tabular-nums text-fg-muted">
                  +1
                </td>
                <td className="px-3 py-2 text-right tabular-nums text-fg-muted">
                  +2
                </td>
              </tr>
              <tr className="border-b border-border last:border-0">
                <td className="px-3 py-2 text-left text-fg">Loss</td>
                <td className="px-3 py-2 text-right tabular-nums text-fg-muted">
                  0
                </td>
                <td className="px-3 py-2 text-right tabular-nums text-fg-muted">
                  −1
                </td>
              </tr>
              <tr className="border-b border-border last:border-0">
                <td className="px-3 py-2 text-left text-fg">
                  Push (line lands exactly)
                </td>
                <td className="px-3 py-2 text-right tabular-nums text-fg-muted">
                  0
                </td>
                <td className="px-3 py-2 text-right tabular-nums text-fg-muted">
                  0
                </td>
              </tr>
              <tr className="border-b border-border last:border-0">
                <td className="px-3 py-2 text-left text-fg">
                  Not gradeable / not applicable
                </td>
                <td className="px-3 py-2 text-right tabular-nums text-fg-muted">
                  0
                </td>
                <td className="px-3 py-2 text-right tabular-nums text-fg-muted">
                  0
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="text-xs text-fg-muted">
          A "push" is when the result lands exactly on the line (no win, no
          loss). A spread pick on a true pick'em game (no favorite) can't be
          graded as a cover, so it scores 0. A game that isn't final yet scores 0
          until it finishes. Misc picks are scored by the admin's decision
          (correct/incorrect + points).
        </p>
      </details>

      <details className="space-y-3 rounded-lg border border-border bg-surface p-4">
        <summary className="cursor-pointer text-lg font-bold text-fg">
          Standings
        </summary>
        <p className="text-sm text-fg-muted">
          Your weekly score is the sum of your picks that week; the season
          standings rank everyone by total points across all weeks. Ties share a
          rank.
        </p>
      </details>

      <section className="space-y-3">
        <h2 className="text-lg font-bold">Using the Discord bot</h2>

        <details
          open
          className="space-y-3 rounded-lg border border-border bg-surface p-4"
        >
          <summary className="cursor-pointer font-bold text-fg">
            How do I talk to the bot?
          </summary>
          <ul className="list-disc space-y-1 pl-5 text-sm text-fg-muted">
            <li>
              @mention the bot in a server channel and ask a plain-English
              question — it replies publicly, in one line.
            </li>
            <li>
              Server channels only for now — it doesn't answer DMs yet.
            </li>
            <li>
              There's a short per-user cooldown (about 10 seconds) so it can't be
              spammed.
            </li>
            <li>It never @-pings anyone in its replies.</li>
            <li>
              A couple of slash commands round it out:{" "}
              <span className="font-semibold">/register</span> to link your
              account and <span className="font-semibold">/reset-password</span>{" "}
              if you're locked out.
            </li>
          </ul>
        </details>

        <details className="space-y-3 rounded-lg border border-border bg-surface p-4">
          <summary className="cursor-pointer font-bold text-fg">
            What can I ask it?
          </summary>
          <p className="text-sm text-fg-muted">
            Ask in plain English — here's what it can answer, with an example of
            each:
          </p>
          <ul className="list-disc space-y-1 pl-5 text-sm text-fg-muted">
            <li>
              <span className="font-semibold">Your own picks & lock status</span>{" "}
              — "did I get all my picks in?" It only ever reports YOUR own picks,
              never anyone else's.
            </li>
            <li>
              <span className="font-semibold">Standings / who's winning</span> —
              "who's winning the league?"
            </li>
            <li>
              <span className="font-semibold">
                Lines & this week's slate
              </span>{" "}
              — "what's the line on the Chiefs game?" or "show me this week's
              slate".
            </li>
            <li>
              <span className="font-semibold">Scores</span> (final or
              in-progress) — "what's the score of the Vikings game?"
            </li>
            <li>
              <span className="font-semibold">Injuries</span> (name a team) —
              "who's hurt on the Eagles?"
            </li>
            <li>
              <span className="font-semibold">Weather for a game</span> (name a
              team) — "what's the weather for the Bills game?"
            </li>
            <li>
              <span className="font-semibold">News / headlines</span> (a team is
              optional) — "any news on the Cowboys?" or "what's the latest NFL
              news?"
            </li>
            <li>
              <span className="font-semibold">
                Game prediction / who covers
              </span>{" "}
              (name a team) — "who's gonna cover the Chiefs game?" That's the
              bot's own analyst call, not your pick.
            </li>
          </ul>
          <p className="text-sm text-fg-muted">
            Team nicknames work too — "niners", "pack", "boys", "gang green",
            "big blue", "bolts", "fins", "mafia", and more all resolve to the
            right team.
          </p>
        </details>

        <details className="space-y-3 rounded-lg border border-border bg-surface p-4">
          <summary className="cursor-pointer font-bold text-fg">
            What does the bot post on its own?
          </summary>
          <p className="text-sm text-fg-muted">
            Without being asked, the bot announces:
          </p>
          <ul className="list-disc space-y-1 pl-5 text-sm text-fg-muted">
            <li>Weekly picks opening for a new week.</li>
            <li>
              Picks locking for the week — with a bit of per-player lock
              commentary.
            </li>
            <li>When a player completes their card for the week.</li>
            <li>A misc call being made, and later graded.</li>
            <li>Final scores as games finish.</li>
            <li>The end-of-week recap.</li>
            <li>The point-spread freeze (when the lines lock).</li>
          </ul>
        </details>

        <details className="space-y-3 rounded-lg border border-border bg-surface p-4">
          <summary className="cursor-pointer font-bold text-fg">
            Coming soon
          </summary>
          <ul className="list-disc space-y-1 pl-5 text-sm text-fg-muted">
            <li>Live line-movement questions.</li>
            <li>Talking to the bot in DMs.</li>
          </ul>
        </details>
      </section>
    </div>
  );
}
