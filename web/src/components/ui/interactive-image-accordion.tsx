/* eslint-disable @next/next/no-img-element */
"use client";
import React, { useState } from "react";

type Item = {
  id: number;
  title: string;
  imageUrl: string;
};

const accordionItems: Item[] = [
  {
    id: 1,
    title: "Voice Assistant",
    imageUrl: "https://images.unsplash.com/photo-1738003667850-a2fb736e31b3?q=80&w=764&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
  },
  {
    id: 2,
    title: "AI Image Generation",
    imageUrl: "https://plus.unsplash.com/premium_photo-1727009856377-715a2643c327?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NXx8aW1hZ2UlMjBnZW5lcmF0aW9ufGVufDB8fDB8fHww",
  },
  {
    id: 3,
    title: "AI Chatbot + Local RAG",
    imageUrl: "https://media.istockphoto.com/id/1933417108/photo/ai-chatbot-artificial-intelligence-concept.webp?a=1&b=1&s=612x612&w=0&k=20&c=faD707ehv7Nc1HBXtMZYbNNHZTYhHEnULlbrgkRNGNE=",
  },
  {
    id: 4,
    title: "AI Agent",
    imageUrl: "https://plus.unsplash.com/premium_photo-1683121710572-7723bd2e235d?q=80&w=1632&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
  },
  {
    id: 5,
    title: "Visual Understanding",
    imageUrl: "https://images.unsplash.com/photo-1557495876-aa846fcac3c4?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8VmlzdWFsJTIwdW5kZXJzdGFuZGluZ3xlbnwwfHwwfHx8MA%3D%3D",
  },
];

function AccordionItem({
  item,
  isActive,
  onMouseEnter,
}: {
  item: Item;
  isActive: boolean;
  onMouseEnter: () => void;
}) {
  return (
    <div
      className={`relative h-[450px] rounded-2xl overflow-hidden cursor-pointer transition-all duration-700 ease-in-out ${
        isActive ? "w-[400px]" : "w-[60px]"
      }`}
      onMouseEnter={onMouseEnter}
    >
      <img
        src={item.imageUrl}
        alt={item.title}
        className="absolute inset-0 w-full h-full object-cover"
        onError={(e) => {
          const t = e.target as HTMLImageElement;
          t.onerror = null;
          t.src = "https://source.unsplash.com/featured/?abstract,gradient";
        }}
      />
      <div className="absolute inset-0 bg-black/40" />
      <span
        className={`absolute text-white text-lg font-semibold whitespace-nowrap transition-all duration-300 ease-in-out ${
          isActive
            ? "bottom-6 left-1/2 -translate-x-1/2 rotate-0"
            : "w-auto text-left bottom-24 left-1/2 -translate-x-1/2 rotate-90"
        }`}
      >
        {item.title}
      </span>
    </div>
  );
}

export function LandingAccordionItem() {
  const [activeIndex, setActiveIndex] = useState(4);
  const handleItemHover = (index: number) => {
    setActiveIndex(index);
  };
  return (
    <div className="bg-white font-sans">
      <section className="container mx-auto px-4 py-12 md:py-24">
        <div className="flex flex-col md:flex-row items-center justify-between gap-12">
          <div className="w-full md:w-1/2 text-center md:text-left">
            <h2 className="text-4xl md:text-6xl font-bold text-gray-900 leading-tight tracking-tighter">
              Accelerate Gen-AI Tasks on Any Device
            </h2>
            <p className="mt-6 text-lg text-gray-600 max-w-xl mx-auto md:mx-0">
              Build high-performance AI apps on-device without the hassle of model
              compression or edge deployment.
            </p>
            <div className="mt-8">
              <a
                href="#contact"
                className="inline-block bg-gray-900 text-white font-semibold px-8 py-3 rounded-lg shadow-lg hover:bg-gray-800 transition-colors duration-300"
              >
                Contact Us
              </a>
            </div>
          </div>
          <div className="w-full md:w-1/2">
            <div className="flex flex-row items-center justify-center gap-4 overflow-x-auto p-4">
              {accordionItems.map((item, index) => (
                <AccordionItem
                  key={item.id}
                  item={item}
                  isActive={index === activeIndex}
                  onMouseEnter={() => handleItemHover(index)}
                />
              ))}
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
