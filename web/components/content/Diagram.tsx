interface DiagramProps {
  src: string;
  alt?: string;
}

export function Diagram({ src, alt = 'Diagram' }: DiagramProps) {
  return (
    <div className="center">
      <figure className="tikz-diagram">
        <img src={src} alt={alt} />
      </figure>
    </div>
  );
}
