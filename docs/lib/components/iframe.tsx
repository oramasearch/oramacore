"use client";

import { useEffect, useState } from "react";

type IframeProps = {
	src: string;
	title: string;
	width?: string;
	height?: string;
	style?: React.CSSProperties;
};

export function Iframe(props: IframeProps) {
	const [mounted, setMounted] = useState(false);

	useEffect(() => {
		setMounted(true);
	}, []);

	if (!mounted) return null;

	return (
		<iframe
			src={props.src}
			title={props.title}
			style={{ width: props.width ?? "100%", height: props.height ?? "400px" }}
			loading="lazy"
			frameBorder="0"
		/>
	);
}
