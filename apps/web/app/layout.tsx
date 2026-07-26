import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SquatSpot | AI Squat Form Feedback",
  description:
    "Upload a squat video and receive private, focused feedback on your form.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
