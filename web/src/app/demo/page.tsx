"use client";
import { OceanHero } from "@/components/ui/aurora-hero-bg-2";

export default function AuroraHeroDemo() {
  return (
    <OceanHero
      title="Transform Your Vision"
      description="Create stunning digital experiences with modern design and smooth animations"
      primaryAction={{
        label: "Get Started",
        onClick: () => console.log("Primary action clicked"),
      }}
      secondaryAction={{
        label: "Learn More",
        onClick: () => console.log("Secondary action clicked"),
      }}
    />
  );
}
