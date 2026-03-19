"use client";
import { useRouter } from "next/navigation";
import { OceanHero } from "@/components/ui/aurora-hero-bg-2";
import { LandingAccordionItem } from "@/components/ui/interactive-image-accordion";

export default function Home() {
  const router = useRouter();
  return (
    <>
      <OceanHero
        title="VideoSceneRAG"
        description="AI-powered video scene understanding. Explore scenes, transcripts, and insights with a beautiful landing experience."
        primaryAction={{
          label: "Get Started",
          onClick: () => {
            router.push("/upload");
          },
        }}
        secondaryAction={{
          label: "Learn More",
          onClick: () => {
            window.open("https://github.com/your-repo/VideoSceneRAG", "_blank");
          },
        }}
      />
      <LandingAccordionItem />
    </>
  );
}
