/**
 * Shell layout: the shared chrome every guarded screen inherits.
 *
 * Order: loud DemoBanner (top) -> main Header -> slim ContextBar -> a centered,
 * max-width, padded <main> rendering the routed page via <Outlet/>.
 */
import { Outlet } from "react-router-dom";

import ContextBar from "./ContextBar";
import DemoBanner from "./DemoBanner";
import Header from "./Header";

export default function AppShell() {
  return (
    <div className="min-h-screen bg-white">
      <DemoBanner />
      <Header />
      <ContextBar />
      <main className="mx-auto w-full max-w-5xl xl:max-w-7xl 2xl:max-w-[1700px] px-4 sm:px-6 lg:px-8 py-6">
        <Outlet />
      </main>
    </div>
  );
}
