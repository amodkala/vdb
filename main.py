import vdb

if __name__ == '__main__':
    db = vdb.VDB()

    corpus = ['A man is eating food.',
          'The girl is carrying a baby.',
          'A man is riding a horse.',
          'A woman is playing violin.',
          'Two men pushed carts through the woods.',
          'A man is riding a white horse on an enclosed ground.',
          'A monkey is playing drums.',
          'A man is eating a piece of bread.',
          'A cheetah is running behind its prey.'
          ]

    db.write(corpus)

    print(db.search(['A man is eating pasta.'], k = 3))
