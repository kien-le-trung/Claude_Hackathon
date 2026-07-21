import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SquatSpot",
  description: "Upload a squat video and receive pose-comparison feedback.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
