interface RefProps {
  to: string;
  label?: string;
}

export function Ref({ to, label }: RefProps) {
  return <a href={`#${to}`}>{label || to}</a>;
}
