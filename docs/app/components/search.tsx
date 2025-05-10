import { OramaSearchBox } from "@orama/react-components";
import { CollectionManager } from "@orama/core";
import type { SharedProps } from 'fumadocs-ui/components/dialog/search';

export default function Search(props: SharedProps) {
  const clientInstance = new CollectionManager({
    url: 'https://staging.collections.orama.com',
    collectionID: 'a6qpvbj0nrfnabjt8rrbztsx',
    readAPIKey: '9lZ5AZViyRSVv3rTf8i25lPlJY2HH2wK',
  })
  return <OramaSearchBox {...props} oramaCoreClientInstance={clientInstance} />;

}
