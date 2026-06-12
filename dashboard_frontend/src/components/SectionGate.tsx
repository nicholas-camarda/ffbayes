import type { ReactNode } from 'react';

export interface GatedSection {
  available?: boolean;
  status?: string;
  reason?: string;
  reason_unavailable?: string;
}

export function SectionGate(props: {
  section: GatedSection | undefined | null;
  title: string;
  children: ReactNode;
}) {
  const { section, title, children } = props;
  if (section && section.available === true) {
    return <>{children}</>;
  }
  const reason =
    section?.reason_unavailable || section?.reason || 'This section is not available for this board.';
  return (
    <section className="section-unavailable" aria-label={`${title} unavailable`}>
      <h3>{title}</h3>
      <p>{reason}</p>
    </section>
  );
}
