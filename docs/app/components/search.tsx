import { OramaSearchBox } from "@orama/react-components";
import { CollectionManager } from "@orama/core";
import type { SharedProps } from 'fumadocs-ui/components/dialog/search';

export default function Search(props: SharedProps) {
  const clientInstance = new CollectionManager({
    url: 'https://oramacore.orama.foo',
    collectionID: 'cxlenmho72jp3qpbdphbmfdn',
    readAPIKey: 'caTS1G81uC8uBoWICSQYzmGjGVBCqxrf',
  })
  return <OramaSearchBox {...props} oramaCoreClientInstance={clientInstance} />;

}
