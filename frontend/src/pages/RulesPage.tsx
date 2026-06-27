/**
 * Rules page — the single readable reference for how the pick'em works.
 *
 * Renders INSIDE AppShell (which is inside RequireAuth), so this is CONTENT
 * ONLY: no shell, header, nav, or auth guard. Pure static JSX — no hooks, no
 * data fetching, no API/lib imports, no new deps. The content is authoritative
 * (code-derived from scoring.py, pick_validation.py, pick_window.py, odds.py,
 * and PROJECT.md) and is rendered faithfully, not invented or altered.
 */
export default function RulesPage() {
  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-bold">Rules</h1>
        <p className="mt-1 text-sm text-gray-500">
          Each NFL week, make your picks against that week's games. Picks are
          graded against the betting line and the final scores, and roll up into
          weekly and season-long standings.
        </p>
      </header>

      <section className="space-y-3 rounded-lg border border-gray-200 bg-white p-4">
        <h2 className="text-lg font-bold">Your weekly picks</h2>
        <p className="text-sm text-gray-700">
          Four bet types, one of each per week.
        </p>
        <ul className="list-disc space-y-1 pl-5 text-sm text-gray-700">
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
        <ul className="list-disc space-y-1 pl-5 text-sm text-gray-700">
          <li>
            <span className="font-semibold">Mortal Lock</span> — one optional
            higher-stakes pick per week. It's a 5th pick on any game/type of your
            choosing; a winning mortal lock is worth more, but a losing one costs
            you (see Scoring).
          </li>
          <li>
            <span className="font-semibold">Misc</span> — one optional free-text
            prediction per week tied to a game (e.g. "Mahomes throws for 350+
            yards"). It's graded by an admin, not automatically.
          </li>
        </ul>
      </section>

      <section className="space-y-3 rounded-lg border border-gray-200 bg-white p-4">
        <h2 className="text-lg font-bold">Roster rules</h2>
        <ul className="list-disc space-y-1 pl-5 text-sm text-gray-700">
          <li>At most one base pick of each of the four types per week.</li>
          <li>The mortal lock is the only same-type duplicate allowed.</li>
          <li>
            You can't pick both sides of the same line on the same game (Favorite
            cover AND Underdog cover, or Over AND Under).
          </li>
          <li>A Misc is never a mortal lock.</li>
        </ul>
      </section>

      <section className="space-y-3 rounded-lg border border-gray-200 bg-white p-4">
        <h2 className="text-lg font-bold">The pick window</h2>
        <ul className="list-disc space-y-1 pl-5 text-sm text-gray-700">
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
            Picks are graded against the line as it was when picks locked (the
            frozen line), plus the final scores.
          </li>
        </ul>
      </section>

      {/* Point values below are documentation of the backend scoring engine —
          keep in sync with app/services/scoring.py `_points_for` if it changes. */}
      <section className="space-y-3 rounded-lg border border-gray-200 bg-white p-4">
        <h2 className="text-lg font-bold">Scoring</h2>
        <div className="overflow-x-auto rounded-lg border border-gray-200 bg-white">
          <table className="min-w-full border-collapse text-sm">
            <thead>
              <tr className="border-b border-gray-200 text-gray-600">
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
              <tr className="border-b border-gray-100 last:border-0">
                <td className="px-3 py-2 text-left text-gray-800">Win</td>
                <td className="px-3 py-2 text-right tabular-nums text-gray-700">
                  +1
                </td>
                <td className="px-3 py-2 text-right tabular-nums text-gray-700">
                  +2
                </td>
              </tr>
              <tr className="border-b border-gray-100 last:border-0">
                <td className="px-3 py-2 text-left text-gray-800">Loss</td>
                <td className="px-3 py-2 text-right tabular-nums text-gray-700">
                  0
                </td>
                <td className="px-3 py-2 text-right tabular-nums text-gray-700">
                  −1
                </td>
              </tr>
              <tr className="border-b border-gray-100 last:border-0">
                <td className="px-3 py-2 text-left text-gray-800">
                  Push (line lands exactly)
                </td>
                <td className="px-3 py-2 text-right tabular-nums text-gray-700">
                  0
                </td>
                <td className="px-3 py-2 text-right tabular-nums text-gray-700">
                  0
                </td>
              </tr>
              <tr className="border-b border-gray-100 last:border-0">
                <td className="px-3 py-2 text-left text-gray-800">
                  Not gradeable / not applicable
                </td>
                <td className="px-3 py-2 text-right tabular-nums text-gray-700">
                  0
                </td>
                <td className="px-3 py-2 text-right tabular-nums text-gray-700">
                  0
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="text-xs text-gray-500">
          A "push" is when the result lands exactly on the line (no win, no
          loss). A spread pick on a true pick'em game (no favorite) can't be
          graded as a cover, so it scores 0. A game that isn't final yet scores 0
          until it finishes. Misc picks are scored by the admin's decision
          (correct/incorrect + points).
        </p>
      </section>

      <section className="space-y-3 rounded-lg border border-gray-200 bg-white p-4">
        <h2 className="text-lg font-bold">Standings</h2>
        <p className="text-sm text-gray-700">
          Your weekly score is the sum of your picks that week; the season
          standings rank everyone by total points across all weeks. Ties share a
          rank.
        </p>
      </section>
    </div>
  );
}
