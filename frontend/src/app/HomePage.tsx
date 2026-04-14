import { useDemoPageQuery } from '../api/hooks';
import { FooterCta } from '../features/marketing/FooterCta';
import { HeroSection } from '../features/marketing/HeroSection';
import { IndustriesSection } from '../features/marketing/IndustriesSection';
import { HowItWorksSection } from '../features/marketing/HowItWorksSection';
import { CapabilitiesSection } from '../features/marketing/CapabilitiesSection';
import { ReleaseOverviewSection } from '../features/marketing/ReleaseOverviewSection';
import { DemoTabs } from '../features/demo/DemoTabs';

export function HomePage() {
  const pageQuery = useDemoPageQuery();

  if (pageQuery.isLoading) {
    return (
      <main className="page-status">
        <div className="page-status__card">
          <p className="eyebrow">Loading</p>
          <h1>Preparing the pilot workspace.</h1>
          <p>Fetching the live demo configuration and proof points from the API.</p>
        </div>
      </main>
    );
  }

  if (pageQuery.isError || !pageQuery.data) {
    return (
      <main className="page-status">
        <div className="page-status__card">
          <p className="eyebrow">Frontend Error</p>
          <h1>Could not load the pilot surface.</h1>
          <p>{pageQuery.error instanceof Error ? pageQuery.error.message : 'Unknown error'}</p>
        </div>
      </main>
    );
  }

  const page = pageQuery.data;

  return (
    <main className="page-shell">
      <HeroSection page={page} />
      <IndustriesSection />
      <CapabilitiesSection />
      <div id="proof">
        <ReleaseOverviewSection release={page.release} />
      </div>
      <HowItWorksSection />
      <div id="demo">
        <DemoTabs demos={page.demos} flowsheets={page.flowsheets} release={page.release} />
      </div>
      <FooterCta />
    </main>
  );
}
